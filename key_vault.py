"""Encrypted API-key storage, health tracking, and concurrent round-robin selection."""
from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

import rag_storage as storage

PROVIDER_DEFAULTS = (
    dict(name="openai", adapter="openai", endpoint="https://api.openai.com/v1", model="gpt-4.1-mini", max_tokens=2048, timeout=45, priority=10),
    dict(name="groq", adapter="groq", endpoint="https://api.groq.com/openai/v1", model="openai/gpt-oss-120b", max_tokens=2048, timeout=45, priority=20),
    dict(name="gemini", adapter="gemini", endpoint="", model="gemini-2.5-flash", max_tokens=2048, timeout=45, priority=30),
    dict(name="anthropic", adapter="anthropic", endpoint="", model="claude-sonnet-4-5", max_tokens=2048, timeout=45, priority=40),
    dict(name="openrouter", adapter="openai_compatible", endpoint="https://openrouter.ai/api/v1", model="openai/gpt-4.1-mini", max_tokens=2048, timeout=45, priority=50),
    dict(name="ox_alpha", adapter="openai_compatible", endpoint="https://api.oxalpha.ai/v1", model="gpt-oss-120b", max_tokens=2048, timeout=45, priority=60),
)
SUPPORTED_ADAPTERS = {"openai", "groq", "gemini", "anthropic", "openai_compatible"}
VALID_STATUSES = {"active", "rate_limited", "temporarily_unavailable", "invalid", "disabled"}
ENV_KEYS = {
    "openai": ("OPENAI_API_KEYS", "OPENAI_API_KEY"),
    "groq": ("GROQ_API_KEYS", "GROQ_API_KEY"),
    "gemini": ("GEMINI_API_KEYS", "GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEYS", "ANTHROPIC_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY"),
    "ox_alpha": ("OX_ALPHA_API_KEYS", "OX_ALPHA_API_KEY"),
}


@dataclass(frozen=True)
class KeyCandidate:
    key_id: str
    provider: str
    source: str
    fingerprint: str
    config: dict = field(repr=False)
    api_key: str = field(repr=False)


class KeyVaultError(RuntimeError):
    pass


_LOCK = threading.RLock()
_DB_PATH = storage.DATA_DIR / "key_pool.sqlite3"
_CONN = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=30)
_CONN.execute("PRAGMA journal_mode=WAL")
_CONN.execute("PRAGMA foreign_keys=ON")
_CONN.executescript("""
CREATE TABLE IF NOT EXISTS providers (
    name TEXT PRIMARY KEY, adapter TEXT NOT NULL, endpoint TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL, max_tokens INTEGER NOT NULL, timeout REAL NOT NULL,
    priority INTEGER NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY, owner_type TEXT NOT NULL CHECK(owner_type IN ('user','system')),
    owner_id TEXT NOT NULL DEFAULT '', provider TEXT NOT NULL, encrypted_key BLOB NOT NULL,
    fingerprint TEXT NOT NULL, display_hint TEXT NOT NULL, label TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(owner_type, owner_id, provider, fingerprint),
    FOREIGN KEY(provider) REFERENCES providers(name) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS key_health (
    fingerprint TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'active',
    cooldown_until REAL NOT NULL DEFAULT 0, failure_count INTEGER NOT NULL DEFAULT 0,
    last_error_kind TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS rotation_state (
    provider TEXT PRIMARY KEY, cursor INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS service_keys (
    user_id TEXT NOT NULL, service TEXT NOT NULL, encrypted_key BLOB NOT NULL,
    display_hint TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(user_id, service)
);
""")
for _provider in PROVIDER_DEFAULTS:
    _CONN.execute(
        """INSERT OR IGNORE INTO providers
           (name, adapter, endpoint, model, max_tokens, timeout, priority, enabled)
           VALUES (:name, :adapter, :endpoint, :model, :max_tokens, :timeout, :priority, 1)""",
        _provider,
    )
_CONN.commit()


def _fernet() -> Fernet:
    configured = os.getenv("RAG_KEY_ENCRYPTION_KEY", "").strip()
    if configured:
        try:
            raw = configured.encode("ascii")
            if len(base64.urlsafe_b64decode(raw)) != 32:
                raise ValueError
            return Fernet(raw)
        except Exception:
            derived = base64.urlsafe_b64encode(hashlib.sha256(configured.encode()).digest())
            return Fernet(derived)

    key_file = storage.DATA_DIR / ".key-encryption-secret"
    if not key_file.exists():
        generated = Fernet.generate_key()
        try:
            descriptor = os.open(str(key_file), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(generated)
        except FileExistsError:
            pass
        try:
            os.chmod(key_file, 0o600)
        except OSError:
            pass
    return Fernet(key_file.read_bytes().strip())


_CIPHER = _fernet()


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def fingerprint(api_key: str, provider: str = "") -> str:
    """Return a non-reversible, provider-scoped identifier for a secret."""
    material = f"{provider.strip().lower()}\0{api_key}" if provider else api_key
    return hashlib.sha256(material.encode()).hexdigest()


def mask_key(api_key: str) -> str:
    if len(api_key) < 8:
        return "****"
    return f"{api_key[:3]}-****{api_key[-4:]}"


def _encrypt(api_key: str) -> bytes:
    return _CIPHER.encrypt(api_key.encode())


def _decrypt(token: bytes) -> str:
    try:
        return _CIPHER.decrypt(token).decode()
    except InvalidToken as exc:
        raise KeyVaultError("An encrypted API key cannot be decrypted with the configured encryption key.") from exc


def _migrate_provider_scoped_fingerprints() -> None:
    """Separate health state when the same secret is used with different providers."""
    with _LOCK, _CONN:
        rows = _CONN.execute(
            "SELECT key_id, provider, encrypted_key, fingerprint FROM api_keys"
        ).fetchall()
        for key_id, provider, token, old_digest in rows:
            try:
                new_digest = fingerprint(_decrypt(token), provider)
            except KeyVaultError:
                continue
            if new_digest == old_digest:
                continue
            health = _CONN.execute(
                """SELECT status, cooldown_until, failure_count, last_error_kind, updated_at
                   FROM key_health WHERE fingerprint=?""",
                (old_digest,),
            ).fetchone()
            if health:
                _CONN.execute(
                    """INSERT OR IGNORE INTO key_health
                       (fingerprint, status, cooldown_until, failure_count, last_error_kind, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (new_digest, *health),
                )
            _CONN.execute(
                "UPDATE api_keys SET fingerprint=? WHERE key_id=?",
                (new_digest, key_id),
            )
            if not _CONN.execute(
                "SELECT 1 FROM api_keys WHERE fingerprint=? LIMIT 1", (old_digest,)
            ).fetchone():
                _CONN.execute("DELETE FROM key_health WHERE fingerprint=?", (old_digest,))


_migrate_provider_scoped_fingerprints()


def upsert_provider(name: str, adapter: str, endpoint: str, model: str, max_tokens: int,
                    timeout: float, priority: int, enabled: bool = True) -> None:
    name, adapter = name.strip().lower(), adapter.strip().lower()
    if not name or not model.strip():
        raise ValueError("Provider name and model are required.")
    if not re_safe_name(name):
        raise ValueError("Provider names may contain lowercase letters, numbers, underscores, and hyphens.")
    if adapter not in SUPPORTED_ADAPTERS:
        raise ValueError(f"Unsupported adapter: {adapter}")
    if adapter == "openai_compatible" and not endpoint.startswith(("http://", "https://")):
        raise ValueError("OpenAI-compatible providers require an HTTP(S) endpoint.")
    if not 1 <= int(max_tokens) <= 131072 or not 1 <= float(timeout) <= 300:
        raise ValueError("Maximum tokens or timeout is outside the supported range.")
    with _LOCK, _CONN:
        _CONN.execute("""INSERT INTO providers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET adapter=excluded.adapter, endpoint=excluded.endpoint,
            model=excluded.model, max_tokens=excluded.max_tokens, timeout=excluded.timeout,
            priority=excluded.priority, enabled=excluded.enabled, updated_at=excluded.updated_at""",
            (name, adapter, endpoint.strip(), model.strip(), int(max_tokens), float(timeout),
             int(priority), int(enabled), _now_iso()))


def re_safe_name(name: str) -> bool:
    import re
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,49}", name))


def list_providers(enabled_only: bool = False) -> list[dict]:
    where = "WHERE enabled=1" if enabled_only else ""
    with _LOCK:
        rows = _CONN.execute(f"""SELECT name, adapter, endpoint, model, max_tokens, timeout,
            priority, enabled, updated_at FROM providers {where} ORDER BY priority, name""").fetchall()
    return [dict(name=r[0], adapter=r[1], endpoint=r[2], model=r[3], max_tokens=r[4],
                 timeout=r[5], priority=r[6], enabled=bool(r[7]), updated_at=r[8]) for r in rows]


def provider_config(name: str) -> dict:
    config = next((p for p in list_providers() if p["name"] == name), None)
    if not config:
        raise ValueError(f"Unknown provider: {name}")
    return config


def add_key(owner_type: str, owner_id: str, provider: str, api_key: str, label: str = "") -> str:
    if owner_type not in {"user", "system"}:
        raise ValueError("Invalid key owner type.")
    if owner_type == "user" and not owner_id:
        raise ValueError("A user ID is required for personal keys.")
    provider_config(provider)
    api_key = api_key.strip()
    if len(api_key) < 8 or api_key.startswith(("your_", "change_")):
        raise ValueError("Enter a valid API key.")
    key_id, digest = str(uuid.uuid4()), fingerprint(api_key, provider)
    with _LOCK, _CONN:
        existing = _CONN.execute("""SELECT key_id FROM api_keys WHERE owner_type=? AND owner_id=?
            AND provider=? AND fingerprint=?""", (owner_type, owner_id, provider, digest)).fetchone()
        if existing:
            _CONN.execute("UPDATE api_keys SET encrypted_key=?, display_hint=?, label=?, enabled=1, updated_at=? WHERE key_id=?",
                          (_encrypt(api_key), mask_key(api_key), label.strip()[:80], _now_iso(), existing[0]))
            _CONN.execute("DELETE FROM key_health WHERE fingerprint=?", (digest,))
            return existing[0]
        _CONN.execute("""INSERT INTO api_keys
            (key_id, owner_type, owner_id, provider, encrypted_key, fingerprint, display_hint, label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (key_id, owner_type, owner_id, provider, _encrypt(api_key), digest,
             mask_key(api_key), label.strip()[:80]))
    return key_id


def list_keys(owner_type: str, owner_id: str = "") -> list[dict]:
    with _LOCK:
        rows = _CONN.execute("""SELECT k.key_id, k.provider, k.display_hint, k.label, k.enabled,
            COALESCE(h.status, 'active'), COALESCE(h.cooldown_until, 0),
            COALESCE(h.failure_count, 0), COALESCE(h.last_error_kind, ''), k.updated_at
            FROM api_keys k LEFT JOIN key_health h ON h.fingerprint=k.fingerprint
            WHERE k.owner_type=? AND k.owner_id=? ORDER BY k.created_at, k.rowid""",
            (owner_type, owner_id)).fetchall()
    now = time_now()
    result = []
    for row in rows:
        status = "disabled" if not row[4] else row[5]
        if status in {"rate_limited", "temporarily_unavailable"} and row[6] <= now:
            status = "active"
        result.append(dict(key_id=row[0], provider=row[1], masked_key=row[2], label=row[3],
                           enabled=bool(row[4]), status=status, cooldown_until=row[6],
                           failure_count=row[7], last_error_kind=row[8], updated_at=row[9]))
    return result


def delete_key(key_id: str, owner_type: str | None = None, owner_id: str | None = None) -> bool:
    clauses, values = ["key_id=?"], [key_id]
    if owner_type is not None:
        clauses.append("owner_type=?")
        values.append(owner_type)
    if owner_id is not None:
        clauses.append("owner_id=?")
        values.append(owner_id)
    with _LOCK, _CONN:
        cursor = _CONN.execute(f"DELETE FROM api_keys WHERE {' AND '.join(clauses)}", values)
    return bool(cursor.rowcount)


def set_key_enabled(key_id: str, enabled: bool, owner_type: str | None = None,
                    owner_id: str | None = None) -> bool:
    clauses, values = ["key_id=?"], [key_id]
    if owner_type is not None:
        clauses.append("owner_type=?")
        values.append(owner_type)
    if owner_id is not None:
        clauses.append("owner_id=?")
        values.append(owner_id)
    values = [int(enabled), _now_iso(), *values]
    with _LOCK, _CONN:
        cursor = _CONN.execute(f"UPDATE api_keys SET enabled=?, updated_at=? WHERE {' AND '.join(clauses)}", values)
    return bool(cursor.rowcount)


def revalidate_key(key_id: str) -> None:
    with _LOCK, _CONN:
        row = _CONN.execute("SELECT fingerprint FROM api_keys WHERE key_id=?", (key_id,)).fetchone()
        if row:
            _CONN.execute("DELETE FROM key_health WHERE fingerprint=?", (row[0],))


def save_service_key(user_id: str, service: str, api_key: str) -> None:
    api_key = api_key.strip()
    if not user_id or len(api_key) < 8:
        raise ValueError("Enter a valid service API key.")
    with _LOCK, _CONN:
        _CONN.execute("""INSERT INTO service_keys VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, service) DO UPDATE SET encrypted_key=excluded.encrypted_key,
            display_hint=excluded.display_hint, updated_at=excluded.updated_at""",
            (user_id, service, _encrypt(api_key), mask_key(api_key), _now_iso()))


def get_service_key(user_id: str, service: str) -> str:
    with _LOCK:
        row = _CONN.execute("SELECT encrypted_key FROM service_keys WHERE user_id=? AND service=?",
                            (user_id, service)).fetchone()
    return _decrypt(row[0]) if row else ""


def service_key_status(user_id: str, service: str) -> dict:
    with _LOCK:
        row = _CONN.execute("SELECT display_hint, updated_at FROM service_keys WHERE user_id=? AND service=?",
                            (user_id, service)).fetchone()
    return {"configured": bool(row), "masked_key": row[0] if row else "", "updated_at": row[1] if row else ""}


def delete_service_key(user_id: str, service: str) -> bool:
    with _LOCK, _CONN:
        cursor = _CONN.execute("DELETE FROM service_keys WHERE user_id=? AND service=?", (user_id, service))
    return bool(cursor.rowcount)


def _environment_keys() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    try:
        configured = json.loads(os.getenv("RAG_SYSTEM_KEYS_JSON", "{}") or "{}")
    except json.JSONDecodeError:
        configured = {}
    if isinstance(configured, dict):
        for provider, values in configured.items():
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list):
                result[provider.lower()] = [str(v).strip() for v in values if str(v).strip()]
    for provider, names in ENV_KEYS.items():
        values = result.setdefault(provider, [])
        for name in names:
            raw = os.getenv(name, "")
            values.extend(v.strip() for v in raw.replace("\n", ",").split(",") if v.strip())
        result[provider] = list(dict.fromkeys(v for v in values if not v.startswith(("your_", "change_"))))
    return result


def time_now() -> float:
    import time
    return time.time()


def _health_map(fingerprints: list[str]) -> dict[str, tuple]:
    if not fingerprints:
        return {}
    placeholders = ",".join("?" for _ in fingerprints)
    with _LOCK:
        rows = _CONN.execute(f"SELECT fingerprint, status, cooldown_until FROM key_health WHERE fingerprint IN ({placeholders})",
                             fingerprints).fetchall()
    return {row[0]: (row[1], row[2]) for row in rows}


def _available(candidate: KeyCandidate, health: dict[str, tuple], now: float) -> bool:
    status, cooldown = health.get(candidate.fingerprint, ("active", 0))
    return status == "active" or status in {"rate_limited", "temporarily_unavailable"} and cooldown <= now


def _rotate(provider: str, candidates: list[KeyCandidate]) -> list[KeyCandidate]:
    if len(candidates) < 2:
        return candidates
    with _LOCK:
        _CONN.execute("BEGIN IMMEDIATE")
        try:
            row = _CONN.execute("SELECT cursor FROM rotation_state WHERE provider=?", (provider,)).fetchone()
            cursor = (row[0] if row else 0) % len(candidates)
            _CONN.execute("""INSERT INTO rotation_state(provider, cursor, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET cursor=excluded.cursor, updated_at=excluded.updated_at""",
                (provider, (cursor + 1) % len(candidates), _now_iso()))
            _CONN.commit()
        except Exception:
            _CONN.rollback()
            raise
    return candidates[cursor:] + candidates[:cursor]


def _db_candidates(owner_type: str, owner_id: str, providers: dict[str, dict]) -> list[KeyCandidate]:
    with _LOCK:
        rows = _CONN.execute("""SELECT key_id, provider, encrypted_key, fingerprint FROM api_keys
            WHERE owner_type=? AND owner_id=? AND enabled=1 ORDER BY created_at, rowid""",
            (owner_type, owner_id)).fetchall()
    candidates = []
    for key_id, provider, token, digest in rows:
        if provider not in providers:
            continue
        try:
            secret = _decrypt(token)
        except KeyVaultError:
            mark_fingerprint(digest, "invalid", "decryption_failed", 0)
            continue
        candidates.append(KeyCandidate(key_id, provider, owner_type, digest, providers[provider], secret))
    return candidates


def personal_candidates(user_id: str, provider_filter: str | None = None) -> list[KeyCandidate]:
    configs = {p["name"]: p for p in list_providers(enabled_only=True)}
    order = {p["name"]: (p["priority"], p["name"]) for p in configs.values()}
    now = time_now()
    personal = _db_candidates("user", user_id, configs)
    personal_health = _health_map([c.fingerprint for c in personal])
    personal = [c for c in personal if _available(c, personal_health, now)]
    if provider_filter:
        personal = [c for c in personal if c.provider == provider_filter]
    personal.sort(key=lambda c: order[c.provider])
    return personal


def system_candidates(provider_filter: str | None = None) -> list[KeyCandidate]:
    configs = {p["name"]: p for p in list_providers(enabled_only=True)}
    order = {p["name"]: (p["priority"], p["name"]) for p in configs.values()}
    now = time_now()
    system = _db_candidates("system", "", configs)
    for provider, keys in _environment_keys().items():
        if provider not in configs:
            continue
        for secret in keys:
            digest = fingerprint(secret, provider)
            system.append(KeyCandidate(f"env:{digest[:16]}", provider, "system", digest, configs[provider], secret))
    # Do not attempt a DB key again if the same secret is also configured in the environment.
    deduped = {c.fingerprint: c for c in system}
    system = list(deduped.values())
    health = _health_map([c.fingerprint for c in system])
    system = [c for c in system if _available(c, health, now)]
    if provider_filter:
        system = [c for c in system if c.provider == provider_filter]
    by_provider: dict[str, list[KeyCandidate]] = {}
    for candidate in system:
        by_provider.setdefault(candidate.provider, []).append(candidate)
    ordered_system = []
    for provider in sorted(by_provider, key=lambda name: order[name]):
        ordered_system.extend(_rotate(provider, by_provider[provider]))
    return ordered_system


def request_candidates(user_id: str) -> list[KeyCandidate]:
    """Compatibility helper. Runtime calls select the system phase lazily."""
    return personal_candidates(user_id) + system_candidates()


def mark_fingerprint(digest: str, status: str, error_kind: str = "", cooldown_seconds: int = 0) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid key status: {status}")
    until = time_now() + max(0, cooldown_seconds) if cooldown_seconds else 0
    with _LOCK, _CONN:
        _CONN.execute("""INSERT INTO key_health
            (fingerprint, status, cooldown_until, failure_count, last_error_kind, updated_at)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(fingerprint) DO UPDATE SET status=excluded.status,
            cooldown_until=excluded.cooldown_until, failure_count=key_health.failure_count+1,
            last_error_kind=excluded.last_error_kind, updated_at=excluded.updated_at""",
            (digest, status, until, error_kind[:80], _now_iso()))


def mark_failure(candidate: KeyCandidate, status: str, error_kind: str,
                 cooldown_seconds: int = 0) -> None:
    mark_fingerprint(candidate.fingerprint, status, error_kind, cooldown_seconds)


def mark_success(candidate: KeyCandidate) -> None:
    with _LOCK, _CONN:
        _CONN.execute("DELETE FROM key_health WHERE fingerprint=?", (candidate.fingerprint,))


def pool_summary(user_id: str = "") -> dict:
    providers = list_providers(enabled_only=True)
    enabled_providers = {provider["name"] for provider in providers}
    personal = list_keys("user", user_id) if user_id else []
    system_db = list_keys("system", "")
    env = _environment_keys()
    now = time_now()
    env_fingerprints = [
        fingerprint(secret, provider)
        for provider, values in env.items() if provider in enabled_providers
        for secret in values
    ]
    env_health = _health_map(env_fingerprints)
    available_personal = [
        key for key in personal
        if key["provider"] in enabled_providers and key["enabled"] and key["status"] == "active"
    ]
    available_system = [
        key for key in system_db
        if key["provider"] in enabled_providers and key["enabled"] and key["status"] == "active"
    ]
    available_env = sum(
        1 for provider, values in env.items() if provider in enabled_providers for secret in values
        if _available(
            KeyCandidate("", provider, "system", fingerprint(secret, provider), {}, secret),
            env_health,
            now,
        )
    )
    return {
        "configured": bool(available_personal or available_system or available_env),
        "providers": len(providers),
        "personal_keys": len([k for k in personal if k["enabled"]]),
        "system_keys": len([k for k in system_db if k["enabled"]]) + sum(len(v) for v in env.values()),
    }


def reset_for_tests() -> None:
    with _LOCK, _CONN:
        _CONN.execute("DELETE FROM api_keys")
        _CONN.execute("DELETE FROM key_health")
        _CONN.execute("DELETE FROM rotation_state")
        _CONN.execute("DELETE FROM service_keys")
        _CONN.execute("DELETE FROM providers")
        for provider in PROVIDER_DEFAULTS:
            _CONN.execute("""INSERT INTO providers
                (name, adapter, endpoint, model, max_tokens, timeout, priority, enabled)
                VALUES (:name, :adapter, :endpoint, :model, :max_tokens, :timeout, :priority, 1)""", provider)


def migrate_legacy_plaintext_keys() -> None:
    """Encrypt keys written by older releases, then remove their plaintext copies."""
    if os.getenv("RAG_SKIP_LEGACY_MIGRATION", "").lower() in {"1", "true"}:
        return
    legacy_path = Path(__file__).with_name("user_api_keys.db")
    if not legacy_path.exists() or legacy_path.resolve() == _DB_PATH.resolve():
        return
    legacy = sqlite3.connect(str(legacy_path), timeout=30)
    try:
        tables = {row[0] for row in legacy.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "model_provider_keys" in tables:
            rows = legacy.execute("SELECT slot, provider, model_name, api_key, enabled FROM model_provider_keys").fetchall()
            for slot, provider, model, api_key, enabled in rows:
                if slot != "chat" or not api_key or api_key.startswith(("your_", "change_")):
                    continue
                if provider == "groq" and (model or "").lower() in {"ox-alpha", "ox_alpha"}:
                    provider, model = "ox_alpha", "gpt-oss-120b"
                if provider in {p["name"] for p in list_providers()}:
                    current = provider_config(provider)
                    upsert_provider(provider, current["adapter"], current["endpoint"],
                                    model or current["model"], current["max_tokens"],
                                    current["timeout"], current["priority"], bool(enabled))
                    add_key("system", "", provider, api_key, "Migrated system key")
            legacy.execute("UPDATE model_provider_keys SET api_key='' WHERE api_key<>''")
        if "user_keys" in tables:
            for user_id, groq_key, tavily_key in legacy.execute("SELECT user_id, groq_key, tavily_key FROM user_keys"):
                if groq_key and not groq_key.startswith("your_"):
                    add_key("user", user_id, "groq", groq_key, "Migrated personal key")
                if tavily_key and not tavily_key.startswith("your_"):
                    save_service_key(user_id, "tavily", tavily_key)
            legacy.execute("UPDATE user_keys SET groq_key='', tavily_key='' WHERE groq_key<>'' OR tavily_key<>''")
        legacy.commit()
    finally:
        legacy.close()


migrate_legacy_plaintext_keys()
