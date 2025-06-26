import os
import threading
import requests
import time
from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
from difflib import get_close_matches

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from google_calendar import create_meeting
from dateutil.parser import parse as parse_date
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import Tool


@tool
def generate_meeting_agenda(context: str) -> str:
    """Generates a meeting agenda based on the provided context."""
    prompt = PromptTemplate.from_template(
        "Generate a structured meeting agenda based on the following context:\n\n{context}\n\nBe concise and professional."
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context})
@tool
def schedule_meeting_with_note(details: str) -> str:
    """
    Schedule a calendar meeting using natural language, e.g.:
    'Schedule a 30-minute meeting with Zeina Jarai tomorrow at 2pm about onboarding.'
    """
    try:
        # Fuzzy name match (replace with actual name extraction if needed)
        possible_names = ["Zeina Jarai"]  # Or dynamically parse from 'details'
        attendees = []

        for name in possible_names:
            email = find_user_email(name)
            if email:
                attendees.append(email)

        # You should already have create_meeting() implemented elsewhere
        result = create_meeting(summary=details, attendees=attendees)
        return f"📅 Meeting created: {result.get('htmlLink')}"
    except Exception as e:
        return f"❌ Failed to schedule meeting: {str(e)}"
# Load env vars
load_dotenv()

# Slack & Gemini setup
slack_token = os.getenv("SLACK_BOT_TOKEN")
gemini_api_key = os.getenv("GEMINI_API_KEY")
slack_client = WebClient(token=slack_token)

# Langchain LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)

# Cache Slack users for fuzzy match
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

@tool
def start_slack_call(name: str) -> str:
    """Starts a Slack call with the closest-matching user."""
    user_id = find_user_id(name)
    if not user_id:
        return f"Could not find a user matching '{name}'."
    try:
        # This starts a call; replace with actual API call if required
        slack_client.chat_postMessage(channel=user_id, text="📞 Starting a call with you!")
        return f"Started a call with {name}."
    except SlackApiError as e:
        return f"Failed to start call: {str(e)}"

@tool
def create_slack_reminder(text: str) -> str:
    """Creates a Slack reminder with the given text."""
    try:
        slack_client.chat_postMessage(channel="#general", text=f"📌 Reminder: {text}")
        return f"Reminder set: {text}"
    except SlackApiError as e:
        return f"Failed to set reminder: {str(e)}"

# Initialize Langchain agent
#agent = initialize_agent(
#    tools=[start_slack_call, create_slack_reminder, schedule_meeting_with_note, generate_meeting_agenda],
#    llm=llm,
#    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#    verbose=True,
#)
# Shared thread memories

@tool
def general_query(input: str) -> str:
    """Handles general questions outside the scope of scheduling and Slack operations."""
    return llm.invoke(input)
thread_memories = {}

def get_agent(thread_ts):
    if thread_ts not in thread_memories:
        thread_memories[thread_ts] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    memory = thread_memories[thread_ts]

    return initialize_agent(
        tools=[
            start_slack_call,
            create_slack_reminder,
            schedule_meeting_with_note,
            generate_meeting_agenda,
            general_query  # 🔁 New fallback tool for general requests
        ],
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

    try:
        agent = get_agent(thread_ts)
        result = agent.run(user_input)

        slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=result)

    except Exception as e:
        slack_client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=f"❌ Langchain error: {str(e)}"
        )

@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json

    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    if "event" in data:
        threading.Thread(target=process_event, args=(data,)).start()

    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(port=3000)
