import os
import pickle
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date  
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Configuration
SCOPES = ["https://www.googleapis.com/auth/calendar"]
CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
TOKEN_FILE = "token.pickle"
DEFAULT_TIMEZONE = "Europe/London"


def get_calendar_service():
    """Authenticate and return the Google Calendar API service using OAuth2."""
    creds = None

    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=8080)

        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)

    return build("calendar", "v3", credentials=creds)


def get_primary_calendar_id(service):
    """Get the primary calendar ID for the authenticated account."""
    calendar_list = service.calendarList().list().execute()
    for calendar in calendar_list.get("items", []):
        if calendar.get("primary"):
            return calendar["id"]
    raise ValueError("❌ No primary calendar found.")


def ensure_future_datetime(parsed_dt: datetime) -> datetime:
    """Ensure datetime is in the future; bump 1 week if in past."""
    now = datetime.now()
    return parsed_dt + timedelta(days=7) if parsed_dt < now else parsed_dt


def create_meeting(summary, attendees, start=None, end=None):
    from auth import get_calendar_service  # Or your method of getting credentials
    service = get_calendar_service()
    calendar_id = get_primary_calendar_id(service)

    now = datetime.utcnow()
    default_start = now + timedelta(hours=1)
    default_end = default_start + timedelta(minutes=30)

   # 🔁 Parse strings to datetime if needed
    if isinstance(start, str):
        start = parse_date(start)
    if isinstance(end, str):
        end = parse_date(end)

    start_dt = start or default_start
    end_dt = end or (start_dt + timedelta(minutes=30))  # ✅ ensure end is after start

    event = {
        "summary": summary,
        "start": {
            "dateTime": start_dt.isoformat(),
            "timeZone": DEFAULT_TIMEZONE,
        },
        "end": {
            "dateTime": end_dt.isoformat(),
            "timeZone": DEFAULT_TIMEZONE,
        },
        "attendees": [{"email": email} for email in attendees],
        "conferenceData": {
            "createRequest": {
                "requestId": f"{summary}-{now.timestamp()}",
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        },
    }

    return service.events().insert(
        calendarId=calendar_id,
        body=event,
        conferenceDataVersion=1
    ).execute()