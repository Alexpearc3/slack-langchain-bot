import sqlite3
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from slack import slack_client
from tools.reminders import send_reminder

DB_FILE = "conversation_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create reminders table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            reminder_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            scheduled_time TEXT NOT NULL,
            text TEXT NOT NULL
        )
    """)

    # Create messages table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_ts TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

def save_message(thread_ts: str, role: str, content: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages (thread_ts, role, content)
        VALUES (?, ?, ?)
    """, (thread_ts, role, content))
    conn.commit()
    conn.close()

def load_messages(thread_ts: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content FROM messages
        WHERE thread_ts = ?
        ORDER BY timestamp ASC
    """, (thread_ts,))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

def load_reminders_into_scheduler(scheduler: BackgroundScheduler):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT reminder_id, user_id, scheduled_time, text FROM reminders")
    reminders = cursor.fetchall()
    conn.close()

    now = datetime.now()
    for reminder_id, user_id, scheduled_time, text in reminders:
        run_time = datetime.fromisoformat(scheduled_time)
        if run_time > now:
            scheduler.add_job(
                func=send_reminder,
                trigger="date",
                run_date=run_time,
                args=[reminder_id, text],
                id=str(reminder_id)
            )
        else:
            # Outdated reminder, optionally clean it up here
            pass