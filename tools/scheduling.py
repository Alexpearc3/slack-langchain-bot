from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from dateparser.search import search_dates
import logging
import pytz
import spacy
from google_calendar import create_meeting
from tools.utils import find_user_email, extract_person_names

nlp = spacy.load("en_core_web_sm")

@tool
def schedule_meeting_with_note(details: str) -> str:
    """Schedules a Google Calendar meeting from natural language input."""
    try:
        logging.info(f"[schedule_meeting_with_note] 🔍 Input: {details}")

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

        people = extract_person_names(details) or ["Zeina Jarai"]
        attendees = [find_user_email(name) for name in people if find_user_email(name)]

        if not attendees:
            return "❌ Couldn't find email addresses for attendees."

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
def generate_meeting_agenda(context: str) -> str:
    """Generates a structured meeting agenda based on the given context."""
    prompt = PromptTemplate.from_template(
        "Generate a structured meeting agenda based on the following context:\n\n{context}\n\nBe concise and professional."
    )
    chain = prompt | StrOutputParser()
    return chain.invoke({"context": context})
