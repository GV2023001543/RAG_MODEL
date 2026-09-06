"""Small SQLite store for documents and the user-visible conversation history."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

DATA_DIR = Path(os.getenv("RAG_DATA_DIR", str(Path(__file__).parent / ".rag_data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = DATA_DIR / "workspace.sqlite3"
LOCK = threading.RLock()
CONN = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False, timeout=30)
CONN.execute("PRAGMA journal_mode=WAL")
CONN.executescript("""
CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL DEFAULT 'New conversation',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS documents (
    thread_id TEXT NOT NULL, filename TEXT NOT NULL, content_hash TEXT NOT NULL,
    metadata TEXT NOT NULL, chunks TEXT NOT NULL, PRIMARY KEY (thread_id, filename)
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL,
    role TEXT NOT NULL, content TEXT NOT NULL, sources TEXT NOT NULL DEFAULT '[]'
);
""")
CONN.commit()


def ensure_thread(thread_id: str, user_id: str = "") -> None:
    owner = user_id or thread_id.split("::", 1)[0]
    if not thread_id or not owner or (user_id and not thread_id.startswith(f"{user_id}::")):
        raise ValueError("This conversation does not belong to the current session.")
    with LOCK, CONN:
        CONN.execute("INSERT OR IGNORE INTO threads(thread_id, user_id) VALUES (?, ?)", (thread_id, owner))


def thread_ids(user_id: str) -> list[str]:
    with LOCK:
        return [r[0] for r in CONN.execute(
            "SELECT thread_id FROM threads WHERE user_id=? ORDER BY updated_at, rowid", (user_id,)
        )]


def title(thread_id: str) -> str:
    with LOCK:
        row = CONN.execute("SELECT title FROM threads WHERE thread_id=?", (thread_id,)).fetchone()
    return row[0] if row else ""


def set_title(thread_id: str, value: str) -> None:
    ensure_thread(thread_id)
    with LOCK, CONN:
        CONN.execute("UPDATE threads SET title=?, updated_at=CURRENT_TIMESTAMP WHERE thread_id=?",
                     (value[:100], thread_id))


def documents(thread_id: str) -> list[dict]:
    with LOCK:
        rows = CONN.execute("SELECT metadata, chunks, content_hash FROM documents WHERE thread_id=? ORDER BY rowid",
                            (thread_id,)).fetchall()
    return [{"meta": json.loads(m), "chunks": json.loads(c), "content_hash": h} for m, c, h in rows]


def put_document(thread_id: str, metadata: dict, chunks: list[dict], content_hash: str) -> None:
    ensure_thread(thread_id)
    with LOCK, CONN:
        CONN.execute("""INSERT INTO documents VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(thread_id, filename) DO UPDATE SET content_hash=excluded.content_hash,
            metadata=excluded.metadata, chunks=excluded.chunks""",
            (thread_id, metadata["filename"], content_hash, json.dumps(metadata), json.dumps(chunks)))


def remove_document(thread_id: str, filename: str) -> None:
    with LOCK, CONN:
        CONN.execute("DELETE FROM documents WHERE thread_id=? AND filename=?", (thread_id, filename))


def messages(thread_id: str) -> list[dict]:
    with LOCK:
        rows = CONN.execute("SELECT role, content, sources FROM messages WHERE thread_id=? ORDER BY id",
                            (thread_id,)).fetchall()
    return [{"role": r, "content": c, "sources": json.loads(s)} for r, c, s in rows]


def append_message(thread_id: str, role: str, content: str, sources: list | None = None) -> None:
    ensure_thread(thread_id)
    with LOCK, CONN:
        CONN.execute("INSERT INTO messages(thread_id, role, content, sources) VALUES (?, ?, ?, ?)",
                     (thread_id, role, content, json.dumps(sources or [])))
        CONN.execute("UPDATE threads SET updated_at=CURRENT_TIMESTAMP WHERE thread_id=?", (thread_id,))


def delete_thread(thread_id: str) -> None:
    with LOCK, CONN:
        for table in ("messages", "documents", "threads"):
            CONN.execute(f"DELETE FROM {table} WHERE thread_id=?", (thread_id,))
