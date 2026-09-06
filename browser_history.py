"""Validated browser-local snapshots of user-visible conversation history."""
from __future__ import annotations

import json
import uuid

SCHEMA_VERSION = 1
USER_KEY = "teamdino.user.v1"
MAX_SNAPSHOT_BYTES = 3_500_000
MAX_THREADS = 40
MAX_MESSAGES_PER_THREAD = 250
MAX_MESSAGE_CHARS = 50_000


def history_key(user_id: str) -> str:
    return f"teamdino.history.v{SCHEMA_VERSION}.{user_id}"


def valid_user_id(value) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return ""


def _safe_source(source) -> dict:
    if not isinstance(source, dict):
        return {}
    return {
        "citation": str(source.get("citation", ""))[:40],
        "source": str(source.get("source", ""))[:300],
        "page": source.get("page") if isinstance(source.get("page"), int) else None,
        "text": str(source.get("text", ""))[:2_000],
    }


def _safe_message(message) -> dict:
    if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
        return {}
    content = message.get("content")
    if not isinstance(content, str) or not content:
        return {}
    sources = [_safe_source(source) for source in message.get("sources", [])[:8]]
    return {
        "role": message["role"],
        "content": content[:MAX_MESSAGE_CHARS],
        "sources": [source for source in sources if source],
    }


def build_snapshot(storage, user_id: str) -> str:
    conversations = []
    for thread_id in storage.thread_ids(user_id)[-MAX_THREADS:]:
        messages = [_safe_message(item) for item in storage.messages(thread_id)[-MAX_MESSAGES_PER_THREAD:]]
        conversations.append({
            "thread_id": thread_id,
            "title": storage.title(thread_id)[:100] or "New conversation",
            "messages": [message for message in messages if message],
        })
    payload = {"version": SCHEMA_VERSION, "user_id": user_id, "conversations": conversations}
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) <= MAX_SNAPSHOT_BYTES:
        return encoded
    # Preserve the newest conversations when browser quota is approached.
    while conversations and len(encoded.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
        conversations.pop(0)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return encoded


def parse_snapshot(raw, user_id: str) -> list[dict]:
    if not raw:
        return []
    if isinstance(raw, str):
        if len(raw.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
            return []
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, dict) or raw.get("version") != SCHEMA_VERSION:
        return []
    if valid_user_id(raw.get("user_id")) != user_id:
        return []
    restored = []
    for item in raw.get("conversations", [])[-MAX_THREADS:]:
        if not isinstance(item, dict):
            continue
        thread_id = str(item.get("thread_id", ""))
        if not thread_id.startswith(f"{user_id}::") or not valid_user_id(thread_id.split("::", 1)[1]):
            continue
        messages = [_safe_message(message) for message in item.get("messages", [])[-MAX_MESSAGES_PER_THREAD:]]
        restored.append({
            "thread_id": thread_id,
            "title": str(item.get("title") or "New conversation")[:100],
            "messages": [message for message in messages if message],
        })
    return restored


def restore_snapshot(storage, user_id: str, raw) -> list[str]:
    restored = []
    existing = set(storage.thread_ids(user_id))
    for conversation in parse_snapshot(raw, user_id):
        thread_id = conversation["thread_id"]
        storage.ensure_thread(thread_id, user_id)
        if thread_id not in existing:
            storage.set_title(thread_id, conversation["title"])
        if not storage.messages(thread_id):
            for message in conversation["messages"]:
                storage.append_message(thread_id, message["role"], message["content"], message["sources"])
            if conversation["messages"]:
                restored.append(thread_id)
    return restored
