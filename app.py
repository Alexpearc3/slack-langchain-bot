import os
import time
import threading
import logging
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from slack_sdk.errors import SlackApiError
from slack import slack_client
from vision import model as vision_model
from agents import get_agent
from database import init_db, load_reminders_into_scheduler, save_message, load_messages
from tools.utils import download_image
from tools.reminders import cleanup_orphaned_reminders
# from utils import download_image  # Uncomment if needed

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

slack_token = os.getenv("SLACK_BOT_TOKEN")

# Flask app
app = Flask(__name__)

# In-memory cache
event_cache = {}
thread_context = {}

# APScheduler
scheduler = BackgroundScheduler()
scheduler.start()


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
    files = event_data.get("files", [])

    if files:
        context_key = f"{channel_id}:{thread_ts}"
        thread_context.setdefault(context_key, [])

        message_parts = [{"text": user_input}]
        headers = {"Authorization": f"Bearer {slack_token}"}

        for file in files:
            if file.get("mimetype", "").startswith("image/"):
                try:
                    file_info = slack_client.files_info(file=file["id"])
                    file_url = file_info["file"]["url_private_download"]
                    image_data = download_image(file_url, headers)
                    message_parts.append({"mime_type": file["mimetype"], "data": image_data})
                except Exception as e:
                    slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts,
                                                  text=f"❌ Error downloading image: {str(e)}")
                    return

        thread_context[context_key].append({"role": "user", "parts": message_parts})
        try:
            response = vision_model.generate_content(thread_context[context_key])
            thread_context[context_key].append({"role": "model", "parts": [{"text": response.text}]})
            slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=response.text)
            save_message(thread_ts, "user", user_input)
            save_message(thread_ts, "ai", response.text)

            # 🔁 Reflect and complete follow-up
            try:
                user_id = event["event"].get("user") or event["authorizations"][0].get("user_id")
                recent_msgs = load_messages(thread_ts)[-2:]
                history_snippet = "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])

                reflect_agent = get_agent(thread_ts)
                followup_prompt = (
                    f"The user's Slack ID is <@{user_id}>.\n"
                    "Review the last two messages. If the user’s request (including any image) "
                    "was not fully completed, take the next step to complete it using tools. "
                    "Make sure you set the reminders for that user using their ID."
                )
                full_input = f"{history_snippet}\n\n{followup_prompt}"

                followup_result = reflect_agent.invoke({"input": full_input})
                followup_text = followup_result.get("output") if isinstance(followup_result, dict) else str(followup_result)

                if followup_text.strip():
                    slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=followup_text)

            except Exception as e:
                slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts,
                                              text=f"❌ Reflection error: {str(e)}")

        except Exception as e:
            slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts,
                                          text=f"❌ Gemini error: {str(e)}")

    else:
        try:
            agent = get_agent(thread_ts)
            user_id = event["event"].get("user") or event["authorizations"][0].get("user_id")
            enhanced_input = f"user_id={user_id}; text={user_input}"

            result = agent.invoke(enhanced_input)
            result_text = result.get("output") if isinstance(result, dict) else str(result)

            save_message(thread_ts, "user", user_input)
            save_message(thread_ts, "ai", result_text)

            slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=result_text)
        except Exception as e:
            slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts,
                                          text=f"❌ Langchain error: {str(e)}")


@app.route("/slack/events", methods=["POST"])
def slack_events():
    data = request.json
    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})
    if "event" in data:
        threading.Thread(target=process_event, args=(data,)).start()
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    try:
        print("🚀 Initializing database...")
        init_db()
        print("📚 Database ready.")

        print("🔁 Loading scheduled reminders into APScheduler...")
        load_reminders_into_scheduler(scheduler)

        print("🧼 Running initial cleanup of orphaned reminders...")
        cleanup_orphaned_reminders("startup")

        print("⚡ Launching Slack bot server on port 3000...")
        app.run(host="0.0.0.0", port=3000)
    except Exception as e:
        print(f"❌ Failed to start the app: {str(e)}")
