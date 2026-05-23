import json
import os
import uuid

FILE = "history.json"

def load_history():
    if not os.path.exists(FILE):
        return [_new_session()]
    try:
        return json.load(open(FILE))
    except:
        return [_new_session()]

def save_history(sessions):
    with open(FILE, "w") as f:
        json.dump(sessions, f, indent=2)

def _new_session():
    return {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "messages": [],
        "company": "Google",
        "progress": {}
    }