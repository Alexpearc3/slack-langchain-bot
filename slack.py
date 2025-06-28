import os
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from difflib import get_close_matches
from dotenv import load_dotenv

# Load environment variables (if not already loaded)
load_dotenv()

# Set up the Slack WebClient
slack_token = os.getenv("SLACK_BOT_TOKEN")
slack_client = WebClient(token=slack_token)

# Cache Slack users to help with fuzzy matching
user_cache = {}

def refresh_user_cache():
    global user_cache
    try:
        response = slack_client.users_list()
        user_cache = {
            user["profile"].get("real_name", ""): user["id"]
            for user in response["members"]
            if not user.get("deleted")
        }
        logging.info("✅ Refreshed Slack user cache.")
    except SlackApiError as e:
        logging.error(f"❌ Failed to refresh Slack user cache: {e.response['error']}")

def find_user_id(name_fragment: str) -> str | None:
    matches = get_close_matches(name_fragment, user_cache.keys(), n=1, cutoff=0.5)
    if matches:
        return user_cache[matches[0]]
    return None

def find_user_email(name_fragment: str) -> str | None:
    user_id = find_user_id(name_fragment)
    if user_id:
        try:
            user_info = slack_client.users_info(user=user_id)
            return user_info["user"]["profile"].get("email")
        except SlackApiError as e:
            logging.warning(f"⚠️ Failed to get user info for {user_id}: {e}")
    return None

def get_channel_id(channel_name: str) -> str | None:
    try:
        response = slack_client.conversations_list()
        for channel in response["channels"]:
            if channel["name"] == channel_name:
                return channel["id"]
    except SlackApiError as e:
        logging.error(f"❌ Failed to fetch channels: {e}")
    return None
