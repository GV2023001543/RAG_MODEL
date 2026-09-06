import json
import uuid

import browser_history
import rag_storage as storage


def make_thread():
    user_id = str(uuid.uuid4())
    thread_id = f"{user_id}::{uuid.uuid4()}"
    storage.ensure_thread(thread_id, user_id)
    return user_id, thread_id


def test_snapshot_restores_missing_conversation_messages_and_sources():
    user_id, thread_id = make_thread()
    storage.set_title(thread_id, "Saved locally")
    storage.append_message(thread_id, "user", "What is retrieval?")
    storage.append_message(thread_id, "assistant", "A grounded answer", [{
        "citation": "1", "source": "notes.pdf", "page": 2, "text": "Relevant passage"
    }])
    snapshot = browser_history.build_snapshot(storage, user_id)
    storage.delete_thread(thread_id)

    assert browser_history.restore_snapshot(storage, user_id, snapshot) == [thread_id]
    assert storage.title(thread_id) == "Saved locally"
    assert storage.messages(thread_id)[1]["sources"][0]["source"] == "notes.pdf"


def test_restore_never_duplicates_existing_messages():
    user_id, thread_id = make_thread()
    storage.append_message(thread_id, "user", "Only once")
    snapshot = browser_history.build_snapshot(storage, user_id)

    assert browser_history.restore_snapshot(storage, user_id, snapshot) == []
    assert len(storage.messages(thread_id)) == 1


def test_snapshot_rejects_another_user_and_malformed_thread_ids():
    user_id = str(uuid.uuid4())
    other_user = str(uuid.uuid4())
    payload = json.dumps({
        "version": 1,
        "user_id": other_user,
        "conversations": [{"thread_id": f"{other_user}::{uuid.uuid4()}", "messages": []}],
    })
    assert browser_history.parse_snapshot(payload, user_id) == []

    malformed = json.dumps({
        "version": 1, "user_id": user_id,
        "conversations": [{"thread_id": f"{user_id}::not-a-uuid", "messages": []}],
    })
    assert browser_history.parse_snapshot(malformed, user_id) == []


def test_snapshot_strips_unsupported_roles_and_limits_content():
    user_id, thread_id = make_thread()
    storage.append_message(thread_id, "system", "hidden")
    storage.append_message(thread_id, "user", "x" * (browser_history.MAX_MESSAGE_CHARS + 50))
    parsed = browser_history.parse_snapshot(browser_history.build_snapshot(storage, user_id), user_id)

    assert len(parsed[0]["messages"]) == 1
    assert parsed[0]["messages"][0]["role"] == "user"
    assert len(parsed[0]["messages"][0]["content"]) == browser_history.MAX_MESSAGE_CHARS
