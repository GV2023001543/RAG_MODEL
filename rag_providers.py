"""Provider-neutral AI calls with encrypted keys, rotation, cooldowns, and failover."""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

import key_vault as vault
from provider_adapters import build_chat_model

logger = logging.getLogger(__name__)
DEFAULT_TEXT_MODEL = "openai/gpt-oss-120b"
DEFAULT_VISION_MODEL = "qwen/qwen3.6-27b"
MODEL_PROVIDER_SLOTS = {
    "chat": {"label": "Chat model", "provider": "groq", "default_model": DEFAULT_TEXT_MODEL},
    "vision": {"label": "Vision / OCR", "provider": "groq", "default_model": DEFAULT_VISION_MODEL},
    "search": {"label": "Web search", "provider": "tavily", "default_model": ""},
}


class MissingAPIKeyError(RuntimeError):
    pass


class ProviderPoolExhausted(RuntimeError):
    def __init__(self, attempts: list[dict]):
        self.attempts = attempts
        providers = list(dict.fromkeys(a["provider"] for a in attempts))
        detail = ", ".join(providers) if providers else "none configured"
        super().__init__(f"All available AI keys and providers failed ({detail}). Check key health in Settings or try again after cooldowns expire.")


@dataclass(frozen=True)
class FailurePolicy:
    status: str
    kind: str
    cooldown_seconds: int = 0
    skip_provider: bool = False


def content_text(content) -> str:
    """Read visible text without exposing provider reasoning blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
            if isinstance(part, str) or isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _resolve_groq_model_name(candidate: str | None) -> str:
    for value in (candidate, os.getenv("GROQ_TEXT_MODEL", "")):
        model = (value or "").strip()
        if model and model.lower() not in {"groq", "ox-alpha", "ox_alpha"}:
            return model
    return DEFAULT_TEXT_MODEL


def _status_code(exc: Exception) -> int | None:
    current = exc
    for _ in range(4):
        for value in (
            getattr(current, "status_code", None),
            getattr(getattr(current, "response", None), "status_code", None),
        ):
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                pass
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if current is None:
            break
    return None


def classify_error(exc: Exception) -> FailurePolicy:
    code = _status_code(exc)
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    quota = any(token in message for token in ("quota exceeded", "insufficient_quota", "billing limit", "credit balance"))
    if quota:
        return FailurePolicy("rate_limited", "quota", int(os.getenv("RAG_QUOTA_COOLDOWN_SECONDS", "900")))
    if code in {401, 403} or any(token in message for token in (
        "invalid api key", "incorrect api key", "authentication failed", "unauthorized", "revoked api key"
    )):
        return FailurePolicy("invalid", "authentication")
    if code == 429 or "rate limit" in message or "ratelimit" in name:
        cooldown = int(os.getenv("RAG_RATE_LIMIT_COOLDOWN_SECONDS", "90"))
        return FailurePolicy("rate_limited", "rate_limit", cooldown)
    if code is not None and code >= 500:
        return FailurePolicy("temporarily_unavailable", f"http_{code}", int(os.getenv("RAG_PROVIDER_COOLDOWN_SECONDS", "30")))
    if any(token in name + " " + message for token in (
        "timeout", "connection", "connecterror", "network", "temporarily unavailable", "service unavailable"
    )):
        return FailurePolicy("temporarily_unavailable", "connection", int(os.getenv("RAG_PROVIDER_COOLDOWN_SECONDS", "30")))
    if code in {400, 404, 422} or any(token in message for token in ("model not found", "unsupported model")):
        return FailurePolicy("active", "request_rejected", 0, skip_provider=True)
    return FailurePolicy("temporarily_unavailable", "provider_error", int(os.getenv("RAG_PROVIDER_COOLDOWN_SECONDS", "30")))


class PooledChatModel:
    """LangChain-compatible wrapper that resolves a fresh key pool per invocation."""
    def __init__(self, user_id: str, tools=None, provider_filter: str | None = None,
                 model_override: str | None = None):
        self.user_id = user_id
        self.tools = list(tools) if tools else None
        self.provider_filter = provider_filter
        self.model_override = model_override

    def bind_tools(self, tools):
        return PooledChatModel(self.user_id, tools, self.provider_filter, self.model_override)

    def invoke(self, messages, config=None):
        attempts: list[dict] = []
        skipped: set[str] = set()
        seen: set[str] = set()
        found = False
        phases = (
            vault.personal_candidates(self.user_id, self.provider_filter),
            None,  # Resolve the system rotation only if personal keys did not succeed.
        )
        for phase in phases:
            candidates = phase if phase is not None else vault.system_candidates(self.provider_filter)
            found = found or bool(candidates)
            for candidate in candidates:
                if candidate.provider in skipped or candidate.fingerprint in seen:
                    continue
                seen.add(candidate.fingerprint)
                provider_config = dict(candidate.config)
                if self.model_override:
                    provider_config["model"] = self.model_override
                try:
                    model = build_chat_model(provider_config, candidate.api_key)
                    if self.tools:
                        model = model.bind_tools(self.tools)
                    response = model.invoke(messages, config=config)
                    vault.mark_success(candidate)
                    logger.info("AI request succeeded provider=%s source=%s", candidate.provider, candidate.source)
                    return response
                except Exception as exc:
                    policy = classify_error(exc)
                    attempts.append({"provider": candidate.provider, "source": candidate.source, "kind": policy.kind})
                    if policy.status != "active":
                        vault.mark_failure(candidate, policy.status, policy.kind, policy.cooldown_seconds)
                    if policy.skip_provider:
                        skipped.add(candidate.provider)
                    logger.warning("AI request failed provider=%s source=%s category=%s",
                                   candidate.provider, candidate.source, policy.kind)
        if not found:
            raise MissingAPIKeyError(
                "No healthy AI key is available. Add a personal key in Settings, or ask an administrator to configure the system key pool."
            )
        raise ProviderPoolExhausted(attempts)


# Backwards-compatible name used by older imports.
FallbackChatModel = PooledChatModel
_REVISION_LOCK = threading.Lock()
_CONFIG_REVISION = 0


def _bump_revision() -> None:
    global _CONFIG_REVISION
    with _REVISION_LOCK:
        _CONFIG_REVISION += 1


def config_revision(user_id: str) -> tuple:
    # The pooled model reads fresh candidates every call. Revision only rebuilds tool bindings.
    return _CONFIG_REVISION, len(vault.list_keys("user", user_id))


def get_llm(user_id: str = ""):
    if not vault.pool_summary(user_id)["configured"]:
        raise MissingAPIKeyError(
            "No healthy AI key is available. Add a personal key in Settings, or configure system keys in the Admin tab."
        )
    return PooledChatModel(user_id)


def get_vision_llm(user_id: str = ""):
    config = vault.provider_config("groq")
    model = os.getenv("GROQ_VISION_MODEL", DEFAULT_VISION_MODEL).strip() or DEFAULT_VISION_MODEL
    if not config["enabled"]:
        raise MissingAPIKeyError("Groq vision/OCR is disabled.")
    return PooledChatModel(user_id, provider_filter="groq", model_override=model)


def chat_model_status(user_id: str) -> dict:
    summary = vault.pool_summary(user_id)
    personal = list_user_provider_keys(user_id)
    provider = next((key["provider"] for key in personal if key["enabled"] and key["status"] == "active"), "")
    if not provider:
        system_providers = {key["provider"] for key in list_system_provider_keys() if key["enabled"] and key["status"] == "active"}
        provider = next((config["name"] for config in vault.list_providers(enabled_only=True)
                         if config["name"] in system_providers or vault._environment_keys().get(config["name"])), "none")
    config = vault.provider_config(provider) if provider != "none" else None
    return {
        **summary,
        "model": config["model"] if config else "Not configured",
        "provider": provider,
    }


def save_search_key(user_id: str, api_key: str) -> None:
    vault.save_service_key(user_id, "tavily", api_key)
    _bump_revision()


def search_key_status(user_id: str) -> dict:
    return vault.service_key_status(user_id, "tavily")


def delete_search_key(user_id: str) -> bool:
    deleted = vault.delete_service_key(user_id, "tavily")
    if deleted:
        _bump_revision()
    return deleted


def get_search_tool(user_id: str):
    from langchain_core.tools import tool
    key = vault.get_service_key(user_id, "tavily") or os.getenv("TAVILY_API_KEY", "").strip()
    if not key or key.startswith("your_"):
        return None

    @tool
    def web_search(query: str) -> dict:
        """Search the live web for current information. Cite result URLs in answers."""
        from tavily import TavilyClient
        result = TavilyClient(api_key=key).search(query=query, max_results=3)
        return {"results": [
            {"title": row.get("title", ""), "url": row.get("url", ""), "content": row.get("content", "")[:2000]}
            for row in result.get("results", [])
        ]}
    return web_search


def save_user_provider_key(user_id: str, provider: str, api_key: str, label: str = "") -> str:
    key_id = vault.add_key("user", user_id, provider, api_key, label)
    _bump_revision()
    return key_id


def list_user_provider_keys(user_id: str) -> list[dict]:
    return vault.list_keys("user", user_id)


def delete_user_provider_key(user_id: str, key_id: str) -> bool:
    deleted = vault.delete_key(key_id, "user", user_id)
    if deleted:
        _bump_revision()
    return deleted


def set_user_provider_key_enabled(user_id: str, key_id: str, enabled: bool) -> bool:
    updated = vault.set_key_enabled(key_id, enabled, "user", user_id)
    if updated:
        _bump_revision()
    return updated


def save_system_provider_key(provider: str, api_key: str, label: str = "") -> str:
    key_id = vault.add_key("system", "", provider, api_key, label)
    _bump_revision()
    return key_id


def list_system_provider_keys() -> list[dict]:
    return vault.list_keys("system", "")


def delete_system_provider_key(key_id: str) -> bool:
    deleted = vault.delete_key(key_id, "system", "")
    if deleted:
        _bump_revision()
    return deleted


def set_system_provider_key_enabled(key_id: str, enabled: bool) -> bool:
    updated = vault.set_key_enabled(key_id, enabled, "system", "")
    if updated:
        _bump_revision()
    return updated


def revalidate_provider_key(key_id: str) -> None:
    vault.revalidate_key(key_id)
    _bump_revision()


def list_provider_configs() -> list[dict]:
    return vault.list_providers()


def save_provider_config(name: str, adapter: str, endpoint: str, model: str,
                         max_tokens: int, timeout: float, priority: int,
                         enabled: bool = True) -> None:
    vault.upsert_provider(name, adapter, endpoint, model, max_tokens, timeout, priority, enabled)
    _bump_revision()


# Compatibility helpers for prior versions and external callers. Secrets are never returned.
def save_user_keys(user_id: str, groq_key: str = "", tavily_key: str = "") -> None:
    if groq_key.strip():
        save_user_provider_key(user_id, "groq", groq_key, "Groq personal key")
    if tavily_key.strip():
        vault.save_service_key(user_id, "tavily", tavily_key)


def load_user_keys(user_id: str) -> dict:
    providers = {item["provider"] for item in list_user_provider_keys(user_id) if item["enabled"]}
    return {"groq_key": "saved" if "groq" in providers else "",
            "tavily_key": "saved" if vault.service_key_status(user_id, "tavily")["configured"] else ""}


def user_has_keys(user_id: str) -> bool:
    return bool(list_user_provider_keys(user_id))


def _resolve_groq_api_key(user_id: str = "", slot: str = "vision") -> str:
    candidate = next((c for c in vault.request_candidates(user_id) if c.provider == "groq"), None)
    return candidate.api_key if candidate else ""


def get_model_provider_config(slot: str) -> dict:
    if slot == "search":
        return {"slot": slot, "label": "Web search", "provider": "tavily", "model_name": "",
                "has_key": bool(os.getenv("TAVILY_API_KEY")), "enabled": True, "key_source": "environment"}
    name = "groq" if slot == "vision" else chat_model_status("")["provider"]
    config = vault.provider_config(name) if name != "none" else vault.provider_config("groq")
    return {"slot": slot, "label": MODEL_PROVIDER_SLOTS[slot]["label"], "provider": name,
            "model_name": DEFAULT_VISION_MODEL if slot == "vision" else config["model"],
            "has_key": any(c.provider == name for c in vault.request_candidates("")),
            "enabled": config["enabled"], "key_source": "pool"}


def list_model_provider_configs() -> list[dict]:
    return [get_model_provider_config(slot) for slot in MODEL_PROVIDER_SLOTS]


def save_model_provider_config(slot: str, provider: str, model_name: str = "", api_key: str = "", enabled: bool = True) -> None:
    if slot not in {"chat", "vision"}:
        raise ValueError("Use environment configuration for web search.")
    current = vault.provider_config(provider)
    save_provider_config(provider, current["adapter"], current["endpoint"], model_name or current["model"],
                         current["max_tokens"], current["timeout"], current["priority"], enabled)
    if api_key.strip():
        save_system_provider_key(provider, api_key, f"{provider.title()} system key")


def clear_model_provider_key(slot: str) -> None:
    provider = get_model_provider_config(slot)["provider"]
    for key in list_system_provider_keys():
        if key["provider"] == provider:
            delete_system_provider_key(key["key_id"])


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, (MissingAPIKeyError, ProviderPoolExhausted, ValueError)):
        return str(exc)
    policy = classify_error(exc)
    if policy.kind == "authentication":
        return "The API key was rejected. Replace or revalidate it in Settings."
    if policy.kind in {"rate_limit", "quota"}:
        return "All currently available keys are rate limited or out of quota. Try again after their cooldowns expire."
    if policy.kind == "connection":
        return "AI providers could not be reached. Check the connection and try again."
    if "recursion" in type(exc).__name__.casefold():
        return "The assistant reached the tool limit. Try a more specific question."
    return "All configured AI routes failed. Review key health and provider configuration in Settings."
