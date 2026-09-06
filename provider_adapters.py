"""Common chat-model adapter registry for every configured AI provider."""
from __future__ import annotations

from collections.abc import Callable

Adapter = Callable[[dict, str], object]
_ADAPTERS: dict[str, Adapter] = {}


def register_adapter(name: str):
    def decorator(builder: Adapter):
        _ADAPTERS[name] = builder
        return builder
    return decorator


def build_chat_model(config: dict, api_key: str):
    adapter = config["adapter"]
    if adapter not in _ADAPTERS:
        raise ValueError(f"No chat adapter is registered for {adapter}.")
    return _ADAPTERS[adapter](config, api_key)


@register_adapter("groq")
def _groq(config: dict, api_key: str):
    from langchain_groq import ChatGroq
    options = dict(
        model=config["model"], api_key=api_key, temperature=0.2,
        timeout=config["timeout"], max_retries=0, max_tokens=config["max_tokens"],
    )
    if config.get("endpoint"):
        options["groq_api_base"] = config["endpoint"]
    return ChatGroq(**options)


def _openai_client(config: dict, api_key: str, compatible: bool):
    from langchain_openai import ChatOpenAI
    options = dict(
        model=config["model"], api_key=api_key, temperature=0.2,
        timeout=config["timeout"], max_retries=0, max_tokens=config["max_tokens"],
    )
    if compatible or config.get("endpoint"):
        options["base_url"] = config["endpoint"]
    if config["name"] == "openrouter":
        options["default_headers"] = {
            "HTTP-Referer": "https://github.com/GV2023001543/RAG_MODEL",
            "X-Title": "TeamDino RAG Workspace",
        }
    return ChatOpenAI(**options)


@register_adapter("openai")
def _openai(config: dict, api_key: str):
    return _openai_client(config, api_key, False)


@register_adapter("openai_compatible")
def _compatible(config: dict, api_key: str):
    return _openai_client(config, api_key, True)


@register_adapter("gemini")
def _gemini(config: dict, api_key: str):
    from langchain_google_genai import ChatGoogleGenerativeAI
    options = dict(
        model=config["model"], api_key=api_key, temperature=0.2,
        request_timeout=config["timeout"], retries=0, max_tokens=config["max_tokens"],
    )
    if config.get("endpoint"):
        options["client_options"] = {"api_endpoint": config["endpoint"]}
    return ChatGoogleGenerativeAI(**options)


@register_adapter("anthropic")
def _anthropic(config: dict, api_key: str):
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise RuntimeError("Install langchain-anthropic to use the Anthropic provider.") from exc
    options = dict(
        model=config["model"], api_key=api_key, temperature=0.2,
        timeout=config["timeout"], max_retries=0, max_tokens=config["max_tokens"],
    )
    if config.get("endpoint"):
        options["base_url"] = config["endpoint"]
    return ChatAnthropic(**options)
