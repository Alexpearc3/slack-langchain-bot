from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from tools.general import handle_general_query, reflect_and_complete
from tools.reminders import (
    create_direct_reminder,
    edit_reminder,
    cancel_reminder,
    show_all_reminders,
    cleanup_orphaned_reminders,
    query_reminders,
)
from tools.scheduling import schedule_meeting_with_note, generate_meeting_agenda
from tools.slack_tools import start_slack_call
from tools.utils import get_current_time, extract_person_names, find_user_id
from memory import load_messages

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

thread_memories = {}

def get_agent(thread_ts):
    if thread_ts not in thread_memories:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        for msg in load_messages(thread_ts):
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            else:
                memory.chat_memory.add_ai_message(msg["content"])

        thread_memories[thread_ts] = memory

    memory = thread_memories[thread_ts]

    from langchain.agents import Tool

    return initialize_agent(
        tools=[
            Tool.from_function(
                reflect_and_complete,
                name="reflect_and_complete",
                description="Reflect on prior conversation and complete any unfinished tasks."
            ),
            Tool.from_function(
                cleanup_orphaned_reminders,
                name="cleanup_orphaned_reminders",
                description="Remove expired or unscheduled reminders from the database."
            ),
            Tool.from_function(
                show_all_reminders,
                name="show_all_reminders",
                description="List all upcoming scheduled reminders."
            ),
            Tool.from_function(
                edit_reminder,
                name="edit_reminder",
                description="Edit a scheduled reminder by ID, with new time and message."
            ),
            Tool.from_function(
                cancel_reminder,
                name="cancel_reminder",
                description="Cancel a scheduled reminder using its ID."
            ),
            Tool.from_function(
                create_direct_reminder,
                name="create_direct_reminder",
                description="Create a reminder for a user. Format: 'user_id=U123; text=remind me in 1 hour to do X'."
            ),
            Tool.from_function(
                get_current_time,
                name="get_current_time",
                description="Get the current local time in a specified city. If no city is given, defaults to UTC."
            ),
            Tool.from_function(
                extract_person_names,
                name="extract_person_names",
                description="Extract person names from text using NLP. Input should be a sentence."
            ),
            Tool.from_function(
                find_user_id,
                name="find_user_id",
                description="Find the Slack user ID for a given display name. Input should be a name string."
            ),
            Tool.from_function(
                start_slack_call,
                name="start_slack_call",
                description="Start a Slack call with a named user."
            ),
            Tool.from_function(
                schedule_meeting_with_note,
                name="schedule_meeting_with_note",
                description="Create a Google Calendar meeting from natural language input."
            ),
            Tool.from_function(
                generate_meeting_agenda,
                name="generate_meeting_agenda",
                description="Generate a structured agenda from a brief meeting context."
            ),
            Tool.from_function(
                handle_general_query,
                name="handle_general_query",
                description="Answer general questions outside of Slack or calendar tasks."
            ),
            Tool.from_function(
                query_reminders,
                name="query_reminders",
                description="Query and filter reminders using natural language, e.g., 'show all reminders from user_id=U123' or 'reminders after 3pm'."
            ),

        ],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )

