import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Configuration
SCOPES = ['https://www.googleapis.com/auth/calendar']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_PICKLE = 'token.pickle'  # Persisted credentials

def get_calendar_service():
    creds = None

    # Load existing credentials from pickle file
    if os.path.exists(TOKEN_PICKLE):
        with open(TOKEN_PICKLE, 'rb') as token:
            creds = pickle.load(token)

    # If credentials are invalid, refresh or login again
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=8080)

        # Save the new credentials
        with open(TOKEN_PICKLE, 'wb') as token:
            pickle.dump(creds, token)

    # Return a service object
    return build('calendar', 'v3', credentials=creds)

# Optional manual trigger
if __name__ == '__main__':
    service = get_calendar_service()
    print("✅ Authenticated and Calendar service initialized.")
