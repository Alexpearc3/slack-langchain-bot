import os
import threading
import time
import requests
from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
from difflib import get_close_matches
import sqlite3
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from google.oauth2 import service_account
import google.generativeai as genai
from google_calendar import create_meeting
from google_calendar import get_calendar_service
import spacy
import dateparser
from datetime import datetime, timedelta
from dateparser.search import search_dates
import pytz
import re
from datetime import datetime
import dateparser
import logging
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler
# Load environment variables
load_dotenv()
slack_token = os.getenv("SLACK_BOT_TOKEN")
gemini_api_key = os.getenv("GEMINI_API_KEY")
nlp = spacy.load("en_core_web_sm")
# Langchain LLM (text-only)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key)

# Gemini direct (for image processing)
genai.configure(api_key=gemini_api_key)
vision_model = genai.GenerativeModel("models/gemini-2.5-flash")

# Slack client
slack_client = WebClient(token=slack_token)
scheduler = BackgroundScheduler()
scheduler.start()
# Cache Slack users for fuzzy name match
user_cache = {}
def refresh_user_cache():
    global user_cache
    result = slack_client.users_list()
    user_cache = {
        user["profile"].get("real_name", ""): user["id"]
        for user in result["members"]
        if not user.get("deleted")
    }
refresh_user_cache()

def extract_entities(text):
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return names

def find_user_email(name_fragment):
    matches = get_close_matches(name_fragment, user_cache.keys(), n=1, cutoff=0.5)
    if matches:
        user_id = user_cache[matches[0]]
        user_info = slack_client.users_info(user=user_id)
        return user_info["user"]["profile"].get("email")
    return None

def find_user_id(name_fragment):
    matches = get_close_matches(name_fragment, user_cache.keys(), n=1, cutoff=0.5)
    if matches:
        return user_cache[matches[0]]
    return None

# Langchain tools

@tool
def get_current_time(city: str) -> str:
    """    
    Use this tool to get the **current local time** in a city. 
    For example: "What time is it in Brighton?" or "Current time in Tokyo".
    """

    try:
        # 1. Get coordinates from city name
        geolocator = Nominatim(user_agent="time_tool")
        location = geolocator.geocode(city)
        if not location:
            return f"❌ Couldn't find location for '{city}'."

        # 2. Get timezone from coordinates
        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lng=location.longitude, lat=location.latitude)
        if not tz_name:
            return f"❌ Couldn't find timezone for '{city}'."

        # 3. Get current time in that timezone
        now = datetime.now(ZoneInfo(tz_name))
        return now.strftime(f"📍 Current time in {city.title()}: %Y-%m-%d %H:%M:%S (%Z)")

    except Exception as e:
        return f"❌ Error getting time: {str(e)}"
    
@tool
def generate_meeting_agenda(context: str) -> str:
    """Generates a structured meeting agenda based on the given context."""
    prompt = PromptTemplate.from_template(
        "Generate a structured meeting agenda based on the following context:\n\n{context}\n\nBe concise and professional."
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context})

def extract_person_names(text: str) -> list:
    """Extract person names using spaCy NER."""
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents if ent.label_ == "PERSON"))

@tool
def schedule_meeting_with_note(details: str) -> str:
    """
    Schedules a calendar meeting from natural language input.
    """
    try:
        logging.info(f"[schedule_meeting_with_note] 🔍 Input: {details}")

        # ⏱ Try to find a datetime string in the full sentence
        parsed_result = search_dates(
            details,
            settings={
                "PREFER_DATES_FROM": "future",
                "RETURN_AS_TIMEZONE_AWARE": True,
                "RELATIVE_BASE": datetime.now()
            }
        )

        if not parsed_result:
            logging.warning("[schedule_meeting_with_note] ❌ No datetime found in text.")
            return "❌ Couldn't parse date/time. Try something like '2025-06-22 14:00'."

        matched_text, parsed_datetime = parsed_result[0]
        logging.info(f"[schedule_meeting_with_note] 📆 Extracted datetime: {parsed_datetime} (from '{matched_text}')")

        parsed_datetime_utc = parsed_datetime.astimezone(pytz.utc)

        # 👥 Extract person names
        people = extract_person_names(details) or ["Zeina Jarai"]
        attendees = [find_user_email(name) for name in people if find_user_email(name)]

        if not attendees:
            return "❌ Couldn't find email addresses for attendees."

        # 🗓 Create the calendar event
        event = create_meeting(
            summary=details,
            attendees=attendees,
            start=parsed_datetime_utc
        )

        return f"📅 Meeting created: {event.get('htmlLink', '(no link)')}"

    except Exception as e:
        logging.exception("[schedule_meeting_with_note] ❌ Exception occurred.")
        return f"❌ Error scheduling meeting: {str(e)}"
    
@tool
def start_slack_call(name: str) -> str:
    """Starts a Slack call by messaging the closest-matching user."""
    user_id = find_user_id(name)
    if not user_id:
        return f"Could not find a user matching '{name}'."
    try:
        slack_client.chat_postMessage(channel=user_id, text="📞 Starting a call with you!")
        return f"Started a call with {name}."
    except SlackApiError as e:
        return f"Failed to start call: {str(e)}"


@tool
def create_slack_reminder(text_and_context: dict) -> str:
    """
    Creates a Slack reminder in the same channel or thread as the original message.
    
    Requires:
    - 'text': The reminder message
    - 'channel_id': Slack channel ID to post in
    - 'thread_ts' (optional): Timestamp of the thread to reply into
    """
    try:
        text = text_and_context.get("text")
        channel_id = text_and_context.get("channel_id")
        thread_ts = text_and_context.get("thread_ts")  # Optional

        if not text or not channel_id:
            return "❌ Missing 'text' or 'channel_id'."

        slack_client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=f"📌 Reminder: {text}"
        )
        return f"Reminder set in <#{channel_id}>: {text}"

    except SlackApiError as e:
        return f"Failed to set reminder: {str(e)}"
# @tool
# def create_direct_reminder(input_str: str) -> str:
#     """
#     Creates a direct (DM) reminder for the user.
#     Input format: 'user_id=U123ABC; text=Remind me in 1 hour to check the dryer'
#     """
#     import re
#     from dateparser import parse

#     match = re.search(r"user_id=(.*?); text=(.*)", input_str)
#     if not match:
#         return "❌ Invalid input format. Use 'user_id=XYZ; text=Your reminder message'"

#     user_id, raw_text = match.groups()
#     raw_text = raw_text.strip()

#     # Try to extract datetime using dateparser
#     parsed_result = search_dates(raw_text, settings={"PREFER_DATES_FROM": "future"})
#     if not parsed_result:
#         return "❌ Couldn't parse a date or time from your message."

#     _, parsed_time = parsed_result[0]
#     reminder_message = re.sub(r"(in|at|after|from)\s+.*?(to|for)?", "", raw_text, flags=re.IGNORECASE).strip()

#     try:
#         slack_client.chat_postMessage(
#             channel=user_id.strip(),
#             text=f"📌 Reminder (set for {parsed_time.strftime('%H:%M')}): {reminder_message}"
#         )
#         return f"✅ Reminder set for you at {parsed_time.strftime('%H:%M')} – {reminder_message}"
#     except SlackApiError as e:
#         return f"❌ Failed to set reminder: {str(e)}"

@tool
def create_direct_reminder(input_str: str) -> str:
    """
    Creates a direct (DM) reminder for the user, stores it in SQLite, and schedules it.
    Input format: 'user_id=U123ABC; text=Remind me in 1 hour to check the dryer'
    """
    import re
    from dateparser.search import search_dates
    from datetime import datetime

    def extract_reminder_details(text: str):
        parsed = search_dates(text, settings={"PREFER_DATES_FROM": "future"})
        if not parsed:
            return None, None
        for matched_str, dt in parsed:
            if dt > datetime.now():
                reminder_time = dt
                reminder_message = text.replace(matched_str, "").strip(" ,.")
                return reminder_time, reminder_message
        return None, None

    match = re.search(r"user_id=(.*?); text=(.*)", input_str)
    if not match:
        return "❌ Invalid input format. Use 'user_id=XYZ; text=Your reminder message'"

    user_id, raw_text = match.groups()
    user_id = user_id.strip()
    raw_text = raw_text.strip()

    # Intelligent time and message extraction
    parsed_time, reminder_message = extract_reminder_details(raw_text)
    if not parsed_time:
        return "❌ Couldn't parse a date or time from your message."
    if not reminder_message:
        return "❌ Failed to extract a clean reminder message."

    try:
        # Store in DB and get reminder_id
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO reminders (user_id, scheduled_time, text) VALUES (?, ?, ?)",
            (user_id, parsed_time.isoformat(), reminder_message)
        )
        reminder_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Schedule reminder
        scheduler.add_job(
            func=send_reminder,
            trigger="date",
            run_date=parsed_time,
            args=[reminder_id, reminder_message],
            id=str(reminder_id)
        )

        # Immediate DM confirmation
        slack_client.chat_postMessage(
            channel=user_id,
            text=f"📌 Reminder (set for {parsed_time.strftime('%Y-%m-%d %H:%M')}): {reminder_message}"
        )

        return f"✅ Reminder set (ID {reminder_id}) for {parsed_time.strftime('%Y-%m-%d %H:%M')} – {reminder_message}"

    except Exception as e:
        return f"❌ Failed to set reminder: {str(e)}"
    
@tool
def edit_reminder(input_str: str) -> str:
    """
    Updates the time and/or content of an existing reminder.
    Input format: 'reminder_id=1; new_time=tomorrow at 5pm; new_text=Check the pasta'
    """
    import re

    match = re.search(r"reminder_id=(\d+);\s*new_time=(.*?);\s*new_text=(.*)", input_str)
    if not match:
        return "❌ Invalid input format. Use 'reminder_id=ID; new_time=TIME; new_text=TEXT'"

    reminder_id, new_time, new_text = match.groups()
    reminder_id = int(reminder_id.strip())
    new_time = new_time.strip()
    new_text = new_text.strip()

    new_dt = dateparser.parse(new_time, settings={"PREFER_DATES_FROM": "future"})
    if not new_dt:
        return f"❌ Could not parse time from '{new_time}'"

    try:
        # Cancel old scheduled job
        scheduler.remove_job(str(reminder_id))

        # Reschedule job
        scheduler.add_job(
            func=send_reminder,
            trigger="date",
            run_date=new_dt,
            args=[reminder_id, new_text],
            id=str(reminder_id),
        )

        # Update DB
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE reminders SET scheduled_time = ?, text = ? WHERE reminder_id = ?",
            (new_dt.isoformat(), new_text, reminder_id)
        )
        conn.commit()
        conn.close()

        return f"✏️ Reminder {reminder_id} updated to '{new_text}' at {new_dt.strftime('%Y-%m-%d %H:%M')}"

    except Exception as e:
        return f"❌ Failed to edit reminder: {str(e)}"
    
@tool
def cancel_reminder(reminder_id: str) -> str:
    """
    Cancels a previously scheduled reminder using its ID.
    """
    try:
        job_id = str(reminder_id).strip().replace("reminder_id=", "")

        # Cancel via APScheduler (ignore if not found)
        try:
            scheduler.remove_job(job_id)
        except Exception:
            pass  # If job doesn't exist, continue

        # Remove from DB regardless
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM reminders WHERE reminder_id = ?", (job_id,))
        conn.commit()
        conn.close()

        return f"🗑️ Reminder {job_id} cancelled."
    except Exception as e:
        return f"❌ Failed to cancel reminder: {str(e)}"
    
@tool("show_all_reminders")
def show_all_reminders(dummy: str = "now") -> str:
    """
    Displays all saved reminders. Input is ignored, but must not be blank.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute("SELECT reminder_id, user_id, scheduled_time, text FROM reminders ORDER BY scheduled_time ASC")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "📭 No reminders found."

        output = ["📋 *Scheduled Reminders:*"]
        for reminder_id, user_id, scheduled_time, text in rows:
            time_fmt = datetime.fromisoformat(scheduled_time).strftime("%Y-%m-%d %H:%M")
            output.append(f"🔔 ID `{reminder_id}` — <@{user_id}> at `{time_fmt}`: _{text}_")

        return "\n".join(output)

    except Exception as e:
        return f"❌ Failed to load reminders: {str(e)}"

    
@tool
def cleanup_orphaned_reminders(dummy: str = "") -> str:
    """
    Cleans up reminders that are either:
    - expired (in the past)
    - have no matching scheduled job in APScheduler
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute("SELECT reminder_id, scheduled_time FROM reminders")
        reminders = cursor.fetchall()

        now = datetime.now()
        removed = []
        skipped = []

        for reminder_id, scheduled_time in reminders:
            try:
                dt = datetime.fromisoformat(scheduled_time)

                # Check for expired or missing scheduler job
                job_exists = scheduler.get_job(str(reminder_id)) is not None

                if dt < now or not job_exists:
                    cursor.execute("DELETE FROM reminders WHERE reminder_id = ?", (reminder_id,))
                    removed.append(reminder_id)
                else:
                    skipped.append(reminder_id)

            except Exception as e:
                print(f"⚠️ Failed to process reminder {reminder_id}: {e}")

        conn.commit()
        conn.close()

        return (
            f"🧹 Cleanup complete:\n"
            f"✅ Removed reminders: {removed or 'None'}\n"
            f"⏭️ Still active: {skipped or 'None'}"
        )

    except Exception as e:
        return f"❌ Cleanup failed: {str(e)}"

@tool
def handle_general_query(input: str) -> str:
    """Answers general-purpose questions outside of Slack or calendar tasks."""
    
    return llm.invoke(input)
from langchain_core.runnables import RunnableLambda
@tool
def reflect_and_complete(input: str = "") -> str:
    """
    Reflect on previous conversation history via memory and complete unfinished actions.
    """
    prompt = PromptTemplate.from_template("{input}")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"input": input})

# Persistent SQLite DB setup
DB_FILE = "conversation_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create table for storing conversation history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            thread_ts TEXT,
            role TEXT,
            content TEXT
        )
    """)

    # Create table for storing reminders
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            reminder_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            scheduled_time TEXT NOT NULL,
            text TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
def load_reminders_into_scheduler():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT reminder_id, user_id, scheduled_time, text FROM reminders")
    reminders = cursor.fetchall()
    conn.close()

    now = datetime.now()

    for reminder_id, user_id, scheduled_time, text in reminders:
        try:
            scheduled_dt = datetime.fromisoformat(scheduled_time)
            if scheduled_dt > now:
                scheduler.add_job(
                    func=send_reminder,
                    trigger="date",
                    run_date=scheduled_dt,
                    args=[reminder_id, text],
                    id=str(reminder_id),
                    replace_existing=True
                )
                print(f"✅ Re-loaded reminder {reminder_id}")
            else:
                print(f"⏭️ Skipped expired reminder {reminder_id}")
        except Exception as e:
            print(f"❌ Failed to reload reminder {reminder_id}: {e}")
def send_reminder(reminder_id, message_text):
    try:
        # Look up user_id from DB
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM reminders WHERE reminder_id = ?", (reminder_id,))
        row = cursor.fetchone()

        if not row:
            print(f"⚠️ Reminder ID {reminder_id} not found in DB.")
            conn.close()
            return

        user_id = row[0]

        # Send reminder via Slack DM
        slack_client.chat_postMessage(
            channel=user_id,
            text=f"⏰ Reminder: {message_text}"
        )
        print(f"✅ Sent reminder {reminder_id} to {user_id}")

        # Delete reminder from DB after sending
        cursor.execute("DELETE FROM reminders WHERE reminder_id = ?", (reminder_id,))
        conn.commit()
        conn.close()

        # Bonus logging
        print(f"🧹 Cleaned up reminder {reminder_id} from DB.")

    except Exception as e:
        print(f"❌ Failed to send reminder {reminder_id}: {str(e)}")

    
def save_message(thread_ts, role, content):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (thread_ts, role, content) VALUES (?, ?, ?)", (thread_ts, role, content))
    conn.commit()
    conn.close()

def load_messages(thread_ts):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages WHERE thread_ts = ?", (thread_ts,))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

# Agent memory setup
thread_memories = {}
def get_agent(thread_ts):
    if thread_ts not in thread_memories:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Load from DB
        for msg in load_messages(thread_ts):
            memory.chat_memory.add_user_message(msg["content"]) if msg["role"] == "user" else memory.chat_memory.add_ai_message(msg["content"])
        
        thread_memories[thread_ts] = memory

    memory = thread_memories[thread_ts]

    return initialize_agent(
        tools=[
            reflect_and_complete,
            cleanup_orphaned_reminders,
            show_all_reminders,
            edit_reminder,
            cancel_reminder,
            create_direct_reminder,
            get_current_time,
            start_slack_call,
            #create_slack_reminder,
            schedule_meeting_with_note,
            generate_meeting_agenda,
            handle_general_query        ],
                llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
    )

# Flask setup
app = Flask(__name__)
thread_context = {}
event_cache = {}

def cleanup_event_cache(ttl=60):
    now = time.time()
    to_delete = [eid for eid, ts in event_cache.items() if now - ts > ttl]
    for eid in to_delete:
        del event_cache[eid]

def download_image(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.content

def process_event(event):
    event_id = event.get("event_id")
    event_data = event["event"]
    if event_id in event_cache:
        return
    event_cache[event_id] = time.time()
    cleanup_event_cache()

    if event_data.get("type") != "app_mention":
        return

    user_input = event_data["text"]
    channel_id = event_data["channel"]
    thread_ts = event_data.get("thread_ts") or event_data["ts"]
    files = event_data.get("files", [])

    if files:
        context_key = f"{channel_id}:{thread_ts}"
        if context_key not in thread_context:
            thread_context[context_key] = []

        message_parts = [{"text": user_input}]
        headers = {"Authorization": f"Bearer {slack_token}"}

        for file in files:
            if file.get("mimetype", "").startswith("image/"):
                try:
                    file_info = slack_client.files_info(file=file["id"])
                    file_url = file_info["file"]["url_private_download"]
                    image_data = download_image(file_url, headers)
                    message_parts.append({
                        "mime_type": file["mimetype"],
                        "data": image_data
                    })
                except Exception as e:
                    slack_client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=f"❌ Error downloading image: {str(e)}"
                    )
                    return

        thread_context[context_key].append({"role": "user", "parts": message_parts})
        try:
            response = vision_model.generate_content(thread_context[context_key])
            thread_context[context_key].append({"role": "model", "parts": [{"text": response.text}]})
            slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=response.text)
            save_message(thread_ts, "user", user_input)   # user's image-related message
            save_message(thread_ts, "ai", response.text)  # Gemini hybrid model's response
            # 🧠 Reflect and complete: manually check past 2 messages and see if task was fulfilled
            try:
                user_id = event["event"].get("user") or event["authorizations"][0].get("user_id")
                recent_msgs = load_messages(thread_ts)[-2:]  # Get last 2 messages from DB
                history_snippet = "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])

                reflect_agent = get_agent(thread_ts)
                followup_prompt = (
                    f"The user's Slack ID is <@{user_id}>.\n"
                    "Review the last two messages. If the user’s request (including any image) "
                    "was not fully completed, take the next step to complete it using tools. "
                    "Make sure you set the reminders for that user using their ID."
                )
                full_input = f"{history_snippet}\n\n{followup_prompt}"

                followup_result = reflect_agent.invoke({"input": full_input})

                # Extract and post response
                followup_text = followup_result.get("output") if isinstance(followup_result, dict) else str(followup_result)
                if followup_text and followup_text.strip():
                    slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=followup_text)

            except Exception as e:
                slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=f"❌ Reflection error: {str(e)}")

        except Exception as e:
            slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=f"❌ Gemini error: {str(e)}")
    else:
        try:
            agent = get_agent(thread_ts)
            user_id = event["event"].get("user") or event["authorizations"][0].get("user_id")
            print(f"🧠 Injecting user ID into input: {user_id}")
            enhanced_input = f"user_id={user_id}; text={user_input}"

            result = agent.invoke(enhanced_input)
            result_text = result.get("output") if isinstance(result, dict) else str(result)

            save_message(thread_ts, "user", user_input)
            save_message(thread_ts, "ai", result_text)

            slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=result_text)
        except Exception as e:
            slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=f"❌ Langchain error: {str(e)}")

def get_channel_id(channel_name: str) -> str | None:
    try:
        response = slack_client.conversations_list()
        for channel in response["channels"]:
            if channel["name"] == channel_name:
                return channel["id"]
    except SlackApiError as e:
        logging.error(f"Failed to fetch channels: {e}")
    return None

def extract_reminder_details(text: str):
    parsed = search_dates(
        text,
        settings={
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": datetime.now(),  # Anchors 'tomorrow', 'in 1 hour', etc.
        }
    )

    if not parsed:
        return None, None

    # Get the first future datetime and its matched phrase
    for phrase, parsed_dt in parsed:
        if parsed_dt > datetime.now():
            # Clean the reminder message by removing the matched date phrase
            reminder_message = text.replace(phrase, "").strip(" ,.")
            return parsed_dt, reminder_message

    return None, None


@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})
    if "event" in data:
        threading.Thread(target=process_event, args=(data,)).start()
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    try:
        print("🚀 Initializing database...")
        init_db()
        print("📚 Database ready.")

        print("🔁 Loading scheduled reminders into APScheduler...")
        load_reminders_into_scheduler()

        print("⚡ Launching Slack bot server...")
        app.run(host="0.0.0.0", port=3000)
    except Exception as e:
        print(f"❌ Failed to start the app: {str(e)}")