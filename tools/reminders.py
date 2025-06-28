import os
import re
import sqlite3
from datetime import datetime
from dateparser.search import search_dates
from apscheduler.schedulers.background import BackgroundScheduler
import google.generativeai as genai
from langchain_core.tools import tool
from slack import slack_client
from langchain_core.prompts import PromptTemplate

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.0-flash")

DB_FILE = "conversation_history.db"
scheduler = BackgroundScheduler()
scheduler.start()
@tool
def query_reminders(instruction: str) -> str:
    """
    Query reminders using natural language.
    Example inputs:
      - 'Show all reminders set by user_id=U123'
      - 'Show reminders after 3pm'
    """
    try:
        # LLM processes instruction and outputs a filter spec
        filter_prompt = PromptTemplate.from_template(
            "Translate this instruction into a JSON reminder query filter:\n{instruction}"
        )
        chain = filter_prompt | llm | StrOutputParser()
        filter_spec = chain.invoke({"instruction": instruction})
        
        # Convert filter_spec JSON to dict and run your query
        filters = json.loads(filter_spec)
        return apply_reminder_filters(filters)  # your SQL or in-memory filter logic
    except Exception as e:
        return f"❌ Failed to query reminders: {str(e)}"
    
def send_reminder(reminder_id, message_text):
    """Send the reminder message to the user and remove it from the database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM reminders WHERE reminder_id = ?", (reminder_id,))
        row = cursor.fetchone()
        if not row:
            print(f"⚠️ Reminder ID {reminder_id} not found in DB.")
            conn.close()
            return
        user_id = row[0]
        slack_client.chat_postMessage(channel=user_id, text=f"⏰ Reminder: {message_text}")
        print(f"✅ Sent reminder {reminder_id} to {user_id}")
        cursor.execute("DELETE FROM reminders WHERE reminder_id = ?", (reminder_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Failed to send reminder {reminder_id}: {str(e)}")

@tool
def handle_general_query(input: str) -> str:
    """Answers general-purpose questions outside of Slack or calendar tasks."""
    try:
        response = model.generate_content(input)
        return response.text
    except Exception as e:
        return f"❌ Gemini error: {str(e)}"

@tool
def reflect_and_complete(input: str = "") -> str:
    """Reflect on previous conversation history via memory and complete unfinished actions."""
    try:
        context = f"Here is the previous conversation:\n\n{input}\n\nWhat needs to be completed?"
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        return f"❌ Reflection error: {str(e)}"

@tool
def create_direct_reminder(input_str: str) -> str:
    """
    Uses Gemini to extract datetime and message, then creates a reminder.
    Format: 'user_id=U123; text=Remind me to do X at 4pm tomorrow'
    """
    try:
        match = re.search(r"user_id=(.*?); text=(.*)", input_str)
        if not match:
            return "❌ Invalid format. Use 'user_id=XYZ; text=Your reminder message'"

        user_id, raw_text = match.groups()
        user_id = user_id.strip()
        raw_text = raw_text.strip()

        # 🌟 Gemini extracts intent
        gemini_prompt = (
            f"Extract the following from this reminder request:\n"
            f"Text: '{raw_text}'\n\n"
            f"Return *only* this JSON:\n"
            f'{{"reminder_text": "...", "datetime": "ISO 8601 format"}}'
        )

        response = model.generate_content(gemini_prompt)
        parsed = response.text

        import json
        data = json.loads(parsed)

        if not data.get("datetime"):
            return "❌ Could not determine a valid time for the reminder."

        parsed_time = datetime.fromisoformat(data["datetime"])
        reminder_text = data["reminder_text"].strip()

        # 🚫 Prevent past scheduling
        now = datetime.now()
        if parsed_time < now:
            parsed_time += timedelta(days=1)

        # ✅ Save to DB and APScheduler
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO reminders (user_id, scheduled_time, text) VALUES (?, ?, ?)",
                       (user_id, parsed_time.isoformat(), reminder_text))
        reminder_id = cursor.lastrowid
        conn.commit()
        conn.close()

        scheduler.add_job(func=send_reminder, trigger="date", run_date=parsed_time,
                          args=[reminder_id, reminder_text], id=str(reminder_id))

        slack_client.chat_postMessage(
            channel=user_id,
            text=f"📌 Reminder (set for {parsed_time.strftime('%Y-%m-%d %H:%M')}): {reminder_text}"
        )
        return f"✅ Reminder set (ID {reminder_id}) for {parsed_time.strftime('%Y-%m-%d %H:%M')} – {reminder_text}"

    except Exception as e:
        return f"❌ Failed to set reminder: {str(e)}"

@tool
def edit_reminder(input_str: str) -> str:
    """Edits an existing reminder's time and text using its ID."""
    match = re.search(r"reminder_id=(\d+);\s*new_time=(.*?);\s*new_text=(.*)", input_str)
    if not match:
        return "❌ Invalid input format. Use 'reminder_id=ID; new_time=TIME; new_text=TEXT'"

    reminder_id, new_time, new_text = match.groups()
    new_dt = search_dates(new_time, settings={"PREFER_DATES_FROM": "future"})
    if not new_dt:
        return f"❌ Could not parse time from '{new_time}'"

    new_dt = new_dt[0][1]  # datetime object

    try:
        scheduler.remove_job(str(reminder_id))
        scheduler.add_job(func=send_reminder, trigger="date", run_date=new_dt,
                          args=[reminder_id, new_text], id=str(reminder_id))

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE reminders SET scheduled_time = ?, text = ? WHERE reminder_id = ?",
                       (new_dt.isoformat(), new_text.strip(), int(reminder_id)))
        conn.commit()
        conn.close()

        return f"✏️ Reminder {reminder_id} updated to '{new_text.strip()}' at {new_dt.strftime('%Y-%m-%d %H:%M')}"
    except Exception as e:
        return f"❌ Failed to edit reminder: {str(e)}"

@tool
def cancel_reminder(reminder_id: str) -> str:
    """Cancels a scheduled reminder and removes it from the database."""
    try:
        job_id = str(reminder_id).strip().replace("reminder_id=", "")
        scheduler.remove_job(job_id)
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM reminders WHERE reminder_id = ?", (job_id,))
        conn.commit()
        conn.close()
        return f"🗑️ Reminder {job_id} cancelled."
    except Exception as e:
        return f"❌ Failed to cancel reminder: {str(e)}"

@tool("show_all_reminders")
def show_all_reminders() -> str:
    """Shows all scheduled reminders from the database."""
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
    """Removes expired or orphaned reminders that are no longer scheduled."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT reminder_id, scheduled_time FROM reminders")
        reminders = cursor.fetchall()

        now = datetime.now()
        removed, skipped = [], []

        for reminder_id, scheduled_time in reminders:
            dt = datetime.fromisoformat(scheduled_time)
            job_exists = scheduler.get_job(str(reminder_id)) is not None
            if dt < now or not job_exists:
                cursor.execute("DELETE FROM reminders WHERE reminder_id = ?", (reminder_id,))
                removed.append(reminder_id)
            else:
                skipped.append(reminder_id)

        conn.commit()
        conn.close()

        return (
            f"🧹 Cleanup complete:\n"
            f"✅ Removed reminders: {removed or 'None'}\n"
            f"⏭️ Still active: {skipped or 'None'}"
        )

    except Exception as e:
        return f"❌ Cleanup failed: {str(e)}"