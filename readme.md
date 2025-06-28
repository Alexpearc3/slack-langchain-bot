# Slack LangChain + Gemini Hybrid Bot

A hybrid Slack bot that combines LangChain agent tools with Google Gemini 2.0 Flash for advanced natural language and image-based Slack interactions.

---

## Features

- **Schedule meetings**  
  `@bot schedule a 30 min meeting with Zeina tomorrow at 1pm`

- **Generate agendas**  
  `@bot what should the agenda be?`

- **Start Slack calls**  WIP
  `@bot start a call with Alex` 

- **Set reminders**  
  `@bot remind everyone about the demo at 3pm`

- **Image analysis (Gemini)**  
  Upload an image and mention the bot with a caption

- **General Queries**  
  `@bot what's the capital of Italy?`

---

## Prerequisites

- Python 3.10+
- Slack App with:
  - Bot token
  - Enabled Event Subscriptions
- Google Cloud project with:
  - Generative Language API enabled (for Gemini)
  - Google Calendar API enabled
  - OAuth client (`credentials.json`)
- (Optional) Cloudflare Tunnel for local webhook exposure

---

## Setup Instructions

### 1. Clone & Install

```bash
git clone https://github.com/Alexpearc3/slack-langchain-bot.git
cd slack-langchain-bot
pip install -r requirements.txt
```

### 2. Create `.env` File

In the project root, create a file named `.env`:

```
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
GEMINI_API_KEY=your-gemini-api-key
```

### 3. Slack App Configuration

In your [Slack App dashboard](https://api.slack.com/apps):

- Enable **Event Subscriptions**
  - Request URL: `https://<your-tunnel>.trycloudflare.com/slack/events`
  - Subscribe to bot events: `app_mention`

- Add **OAuth Scopes** under *OAuth & Permissions*:
  - `chat:write`
  - `users:read`
  - `files:read`

- *(Optional)* Enable **Interactivity** if you plan to support buttons or modals.

### 4. Google Calendar Setup

- Enable the **Google Calendar API** in Google Cloud Console
- Create an OAuth 2.0 client ID (Desktop app type)
- Download `credentials.json` and place it in the root of the project
- On first run, you'll be prompted to authorize access. A `token.pickle` will be created and reused.

---

## Running the Bot

### Local development (test mode only)

Run the bot locally using:

```
python app.py
```

This will start your Flask app on `http://localhost:3000`.

---

### Public Slack interaction (with Cloudflare Tunnel)

To expose your local bot to Slack, run:

```
python setup_tunnel.py
```

This will automatically download and start a Cloudflare tunnel pointing to `http://localhost:3000`.

Once the tunnel is running, copy the generated `https://<your-subdomain>.trycloudflare.com` URL and set your Slack App's **Event Request URL** to:

```
https://<your-subdomain>.trycloudflare.com/slack/events
```

This allows your Slack bot to receive real-time events from Slack while running locally.

---

## Project Structure

- `app.py` — Entry point for the Flask app (LangChain + Gemini agent interface)
- `Hybrid_app.py` — Legacy or alternative app entry (can be deprecated or integrated)
- `agents.py` — LangChain agent definitions and initialization
- `auth.py` — Google OAuth2.0 logic for calendar integration
- `database.py` — SQLite database setup and interaction
- `google_calendar.py` — Google Calendar API integration
- `memory.py` — Conversation memory management (e.g. SQLite-backed memory)
- `slack.py` — Slack event handling and message routing
- `tools/` — Modular LangChain tools used by the agent:
  - `general.py` — Generic utilities and context tools
  - `reminders.py` — Set, edit, and cancel reminders
  - `scheduling.py` — Meeting time parsing and logic
  - `slack_tools.py` — Slack-specific helper tools (e.g. DM, threads)
  - `utils.py` — Shared helper functions
- `vision.py` — Image input handling using Gemini Vision API
- `.env` — Secret keys and tokens (not committed)
- `credentials.json` — Google OAuth client secrets
- `token.pickle` — Auto-generated token from first Google OAuth run
- `requirements.txt` — Python dependency list
- `readme.md` — Project overview and instructions
- `setup_tunnel.py` — Optional script for localtunnel or ngrok setup

---

## To Do

- [ ] Add persistent memory support
- [ ] Add Slack buttons or slash command support
- [ ] Add structured logging (file or cloud-based)
