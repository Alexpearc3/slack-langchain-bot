from langchain.tools import tool
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
from datetime import datetime
from difflib import get_close_matches
from slack import slack_client
import spacy
import requests


def detect_location_by_ip():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        return data.get("city", "UTC")
    except Exception as e:
        print(f"⚠️ Location detection failed: {str(e)}")
        return "UTC"
# Global cache of Slack users
user_cache = {}

def refresh_user_cache():
    """Refresh the Slack user cache for name lookups."""
    global user_cache
    result = slack_client.users_list()
    user_cache = {
        user["profile"].get("real_name", ""): user["id"]
        for user in result["members"]
        if not user.get("deleted")
    }
@tool
def get_current_time(city: str = None) -> str:
    """
    Get the current local time in a city. If no city is provided, auto-detect using IP or fallback to UTC.
    """
    try:
        if not city:
            city = detect_location_by_ip()
            if not city:
                city = "UTC"

        geolocator = Nominatim(user_agent="time_tool")
        location = geolocator.geocode(city)
        if not location:
            return f"❌ Couldn't find location for '{city}'."

        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lng=location.longitude, lat=location.latitude)
        if not tz_name:
            return f"❌ Couldn't find timezone for '{city}'."

        now = datetime.now(ZoneInfo(tz_name))
        return now.strftime(f"📍 Current time in {city.title()}: %Y-%m-%d %H:%M:%S (%Z)")

    except Exception as e:
        return f"❌ Error getting time: {str(e)}"
    
@tool
def extract_person_names(text: str) -> str:
    """
    Extracts person names from text using spaCy.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        names = list(set(ent.text for ent in doc.ents if ent.label_ == "PERSON"))
        return ", ".join(names) if names else "No names found."
    except Exception as e:
        return f"❌ Error extracting names: {str(e)}"

@tool
def find_user_id(name: str) -> str:
    """
    Returns the Slack user ID for the given display name.
    """
    try:
        users = slack_client.users_list().get("members", [])
        name = name.lower()
        for user in users:
            if not user.get("deleted") and name in user.get("real_name", "").lower():
                return user["id"]
        return f"❌ No user found matching '{name}'"
    except Exception as e:
        return f"❌ Error finding user ID: {str(e)}"

# --- Helper utilities (not tools) ---

def find_user_email(name_fragment):
    """Returns Slack user email from partial name match using cached data."""
    matches = get_close_matches(name_fragment, user_cache.keys(), n=1, cutoff=0.5)
    if matches:
        user_id = user_cache[matches[0]]
        user_info = slack_client.users_info(user=user_id)
        return user_info["user"]["profile"].get("email")
    return None

def download_image(url, headers):
    """Download an image from Slack's private URL using auth headers."""
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.content
