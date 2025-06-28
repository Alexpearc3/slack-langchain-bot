from langchain.tools import tool
from slack import slack_client
from slack_sdk.errors import SlackApiError
from tools.utils import find_user_id

@tool
def start_slack_call(name: str) -> str:
    """Starts a Slack call by messaging the closest-matching user."""
    user_id = find_user_id(name)
    if not user_id:
        return f"❌ Could not find a user matching '{name}'."
    try:
        slack_client.chat_postMessage(channel=user_id, text="📞 Starting a call with you!")
        return f"✅ Started a call with {name}."
    except SlackApiError as e:
        return f"❌ Failed to start call: {str(e)}"
