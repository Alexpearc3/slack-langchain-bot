import sqlite3
from langchain.memory import ConversationBufferMemory

DB_FILE = "conversation_history.db"

# Initialize the database schema for messages and reminders
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Table to store conversation history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            thread_ts TEXT,
            role TEXT,
            content TEXT
        )
    """)

    # Table to store reminders
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            reminder_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            scheduled_time TEXT NOT NULL,
            text TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

# Save a single message to DB
def save_message(thread_ts, role, content):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (thread_ts, role, content) VALUES (?, ?, ?)", (thread_ts, role, content))
    conn.commit()
    conn.close()

# Load all messages for a specific thread
def load_messages(thread_ts):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages WHERE thread_ts = ?", (thread_ts,))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

# Optional: in-memory cache of ConversationBufferMemory for agents (not required if done in agents.py)
thread_memories = {}

def get_memory(thread_ts):
    if thread_ts not in thread_memories:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        for msg in load_messages(thread_ts):
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            else:
                memory.chat_memory.add_ai_message(msg["content"])
        thread_memories[thread_ts] = memory
    return thread_memories[thread_ts]
