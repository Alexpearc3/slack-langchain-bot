🤖 Slack LangChain + Gemini Hybrid Bot
This project is a hybrid Slack bot that combines:

✅ Langchain (via Gemini) for structured agent tools like meeting scheduling, call reminders, etc.

🧠 Google Gemini 2.0 Flash for handling image-based input directly

📦 Prerequisites
Ensure you have:

Python 3.10+ installed

Slack App with bot token

Google Cloud project with:

Gemini API key (Generative Language API enabled)

Google Calendar API enabled

credentials.json file downloaded

Cloudflare Tunnel (optional, for local development webhook access)

⚙️ 1. Environment Setup
Install Python dependencies:

powershell
Copy
Edit
pip install -r requirements.txt
Create a .env file in your project root with:

env
Copy
Edit
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
GEMINI_API_KEY=your-gemini-api-key
Place your Google API credentials file:

bash
Copy
Edit
credentials.json  # (client secrets for Calendar)
🧠 2. Slack App Setup
In your Slack App dashboard:

Event Subscriptions:

Enable Events

Request URL: https://your-cloudflare-tunnel-url/slack/events

Subscribe to bot events: app_mention

OAuth & Permissions:

Add scopes:

chat:write

users:read

files:read

Interactivity (optional): Enable if you want buttons in future.

Update the bot name or channel behavior by modifying:

python
Copy
Edit
slack_client.chat_postMessage(channel="#general", ...)  # Change #general as needed
📅 3. Google Calendar Setup
Create project & enable:

Google Calendar API

OAuth 2.0 client ID (for Desktop App)

Download credentials.json to root

On first run, you'll be prompted to authorize access. A token.pickle will be saved for reuse.

🚀 4. Run the Bot (PowerShell)
powershell
Copy
Edit
python hybrid_app.py
For public Slack interaction, use Cloudflare Tunnel:

powershell
Copy
Edit
cloudflared tunnel --url http://localhost:3000
Then set Slack’s Request URL to:

bash
Copy
Edit
https://<your-subdomain>.trycloudflare.com/slack/events
🧪 Features
Feature	Trigger Example
Schedule meeting	“@bot schedule a 30 min meeting with Zeina tomorrow 1pm”
Generate agenda	“@bot what should the agenda be?”
Start Slack call	“@bot start a call with Alex”
Slack reminder	“@bot remind everyone about the demo at 3pm”
Image analysis (Gemini)	Upload an image and mention bot with a caption
General fallback question	“@bot what’s the capital of Italy?”

🛠️ File Overview
File	Purpose
hybrid_app.py	Main Flask + LangChain + Gemini Slack bot
google_calendar.py	Logic for calendar scheduling
.env	Secret tokens (not committed)
credentials.json	Google OAuth client secrets
token.pickle	Auto-generated credentials from first run

🧹 To Do
 Add persistence to conversation history

 Support buttons or slash commands

 Add logging to file or cloud

