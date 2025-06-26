import os
import threading
import requests
import time
from flask import Flask, request, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
import google.generativeai as genai

# Load env variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.0-flash")

# Slack client
slack_token = os.getenv("SLACK_BOT_TOKEN")
slack_client = WebClient(token=slack_token)

# Flask app
app = Flask(__name__)

# Context memory
thread_context = {}

# Deduplication cache
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

    context_key = f"{channel_id}:{thread_ts}"
    if context_key not in thread_context:
        thread_context[context_key] = []

    # Build the message content
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

    # Append user input
    thread_context[context_key].append({
        "role": "user",
        "parts": message_parts
    })

    try:
        response = model.generate_content(thread_context[context_key])
        thread_context[context_key].append({
            "role": "model",
            "parts": [{"text": response.text}]
        })

        slack_client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=response.text
        )

    except Exception as e:
        slack_client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=f"❌ Gemini error: {str(e)}"
        )


@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json

    # Slack URL verification
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    if "event" in data:
        threading.Thread(target=process_event, args=(data,)).start()

    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(port=3000)
