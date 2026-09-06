import logging
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

import key_vault as vault
import rag_providers as providers


def add_system(provider, value, label=""):
    return vault.add_key("system", "", provider, value, label)


def test_keys_are_encrypted_at_rest_and_only_masked_in_public_results():
    secret = "gsk_test_super_secret_123456789"
    key_id = providers.save_user_provider_key("alice", "groq", secret, "Personal")
    encrypted = vault._CONN.execute("SELECT encrypted_key FROM api_keys WHERE key_id=?", (key_id,)).fetchone()[0]
    assert secret.encode() not in encrypted
    public = providers.list_user_provider_keys("alice")[0]
    assert secret not in repr(public)
    assert public["masked_key"].endswith("6789")
    assert "api_key" not in public


def test_user_key_is_always_before_system_pool():
    providers.save_user_provider_key("alice", "anthropic", "user-anthropic-key")
    add_system("openai", "system-openai-key")
    candidates = vault.request_candidates("alice")
    assert candidates[0].source == "user"
    assert candidates[0].provider == "anthropic"
    assert candidates[1].source == "system"


def test_system_keys_rotate_round_robin_without_restarting_at_first():
    for value in ("system-key-A", "system-key-B", "system-key-C"):
        add_system("groq", value)
    starts = [vault.request_candidates("nobody")[0].api_key for _ in range(4)]
    assert starts == ["system-key-A", "system-key-B", "system-key-C", "system-key-A"]


def test_round_robin_is_safe_under_concurrent_requests():
    for value in ("concurrent-A", "concurrent-B", "concurrent-C"):
        add_system("groq", value)
    with ThreadPoolExecutor(max_workers=12) as executor:
        starts = list(executor.map(lambda _: vault.request_candidates("nobody")[0].api_key, range(30)))
    assert {value: starts.count(value) for value in set(starts)} == {
        "concurrent-A": 10, "concurrent-B": 10, "concurrent-C": 10,
    }


def test_personal_success_does_not_advance_system_rotation(monkeypatch):
    providers.save_user_provider_key("alice", "groq", "personal-success-key")
    add_system("groq", "shared-key-A")
    add_system("groq", "shared-key-B")
    used = []

    def build(config, api_key):
        model = Mock()
        used.append(api_key)
        model.invoke.return_value = "ok"
        return model

    monkeypatch.setattr(providers, "build_chat_model", build)
    assert providers.PooledChatModel("alice").invoke("hello") == "ok"
    assert providers.PooledChatModel("nobody").invoke("hello") == "ok"
    assert used == ["personal-success-key", "shared-key-A"]


def test_environment_csv_and_json_keys_join_the_system_pool(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "env-groq-A, env-groq-B\nenv-groq-C")
    monkeypatch.setenv("GROQ_API_KEY", "env-groq-A")
    monkeypatch.setenv("RAG_SYSTEM_KEYS_JSON", json.dumps({"anthropic": ["env-anthropic-A"]}))
    candidates = vault.system_candidates()
    assert [candidate.api_key for candidate in candidates] == [
        "env-groq-A", "env-groq-B", "env-groq-C", "env-anthropic-A"
    ]


def test_disabled_key_and_provider_are_not_reported_as_available():
    key_id = add_system("groq", "disabled-system-key")
    vault.set_key_enabled(key_id, False, "system", "")
    assert not vault.pool_summary()["configured"]
    vault.set_key_enabled(key_id, True, "system", "")
    config = vault.provider_config("groq")
    vault.upsert_provider("groq", config["adapter"], config["endpoint"], config["model"],
                          config["max_tokens"], config["timeout"], config["priority"], False)
    assert not vault.pool_summary()["configured"]
    assert not vault.system_candidates()


def test_health_is_scoped_to_provider_when_secret_is_reused():
    shared_secret = "shared-secret-across-providers"
    add_system("groq", shared_secret)
    add_system("anthropic", shared_secret)
    candidates = vault.system_candidates()
    groq = next(candidate for candidate in candidates if candidate.provider == "groq")
    anthropic = next(candidate for candidate in candidates if candidate.provider == "anthropic")
    assert groq.fingerprint != anthropic.fingerprint
    vault.mark_failure(groq, "invalid", "authentication")
    assert [candidate.provider for candidate in vault.system_candidates()] == ["anthropic"]


def test_provider_priority_controls_cross_provider_fallback_order():
    add_system("groq", "groq-system-key")
    add_system("anthropic", "anthropic-system-key")
    groq = vault.provider_config("groq")
    anthropic = vault.provider_config("anthropic")
    vault.upsert_provider("groq", groq["adapter"], groq["endpoint"], groq["model"], 2048, 45, 90)
    vault.upsert_provider("anthropic", anthropic["adapter"], anthropic["endpoint"], anthropic["model"], 2048, 45, 5)
    assert [candidate.provider for candidate in vault.request_candidates("nobody")] == ["anthropic", "groq"]


def test_rate_limited_key_enters_cooldown_and_next_key_is_selected(monkeypatch):
    monkeypatch.setattr(vault, "time_now", lambda: 1000)
    add_system("groq", "limited-key-A")
    add_system("groq", "healthy-key-B")
    first = vault.request_candidates("nobody")[0]
    vault.mark_failure(first, "rate_limited", "rate_limit", 90)
    assert vault.request_candidates("nobody")[0].api_key == "healthy-key-B"
    monkeypatch.setattr(vault, "time_now", lambda: 1091)
    assert "limited-key-A" in [candidate.api_key for candidate in vault.request_candidates("nobody")]


def test_invalid_key_stays_disabled_until_revalidated():
    key_id = add_system("groq", "invalid-key-A")
    candidate = vault.request_candidates("nobody")[0]
    vault.mark_failure(candidate, "invalid", "authentication")
    assert not vault.request_candidates("nobody")
    vault.revalidate_key(key_id)
    assert vault.request_candidates("nobody")[0].api_key == "invalid-key-A"


class ProviderError(RuntimeError):
    def __init__(self, status_code, message="provider failed"):
        self.status_code = status_code
        super().__init__(message)


def test_user_failure_falls_back_to_system_key(monkeypatch):
    providers.save_user_provider_key("alice", "groq", "personal-key")
    add_system("groq", "shared-key")
    calls = []

    def build(config, api_key):
        model = Mock()
        calls.append(api_key)
        if api_key == "personal-key":
            model.invoke.side_effect = ProviderError(429)
        else:
            model.invoke.return_value = "success"
        return model

    monkeypatch.setattr(providers, "build_chat_model", build)
    assert providers.PooledChatModel("alice").invoke("hello") == "success"
    assert calls == ["personal-key", "shared-key"]
    assert providers.list_user_provider_keys("alice")[0]["status"] == "rate_limited"


def test_exhausted_provider_falls_back_to_next_configured_provider(monkeypatch):
    add_system("openai", "openai-A")
    add_system("openai", "openai-B")
    add_system("groq", "groq-key-A")
    calls = []

    def build(config, api_key):
        model = Mock()
        calls.append((config["name"], api_key))
        if config["name"] == "openai":
            model.invoke.side_effect = ProviderError(429)
        else:
            model.invoke.return_value = "groq success"
        return model

    monkeypatch.setattr(providers, "build_chat_model", build)
    assert providers.PooledChatModel("nobody").invoke("hello") == "groq success"
    assert [provider for provider, _ in calls] == ["openai", "openai", "groq"]


def test_authentication_failure_marks_key_invalid_and_does_not_log_secret(monkeypatch, caplog):
    secret = "secret-invalid-api-key"
    add_system("groq", secret)
    add_system("anthropic", "working-anthropic-key")

    def build(config, api_key):
        model = Mock()
        if config["name"] == "groq":
            model.invoke.side_effect = ProviderError(401, f"invalid API key {api_key}")
        else:
            model.invoke.return_value = "ok"
        return model

    monkeypatch.setattr(providers, "build_chat_model", build)
    with caplog.at_level(logging.WARNING):
        assert providers.PooledChatModel("nobody").invoke("hello") == "ok"
    assert secret not in caplog.text
    assert next(key for key in providers.list_system_provider_keys() if key["provider"] == "groq")["status"] == "invalid"


@pytest.mark.parametrize("error,expected", [
    (ProviderError(429), "rate_limited"),
    (ProviderError(429, "quota exceeded"), "rate_limited"),
    (ProviderError(403, "quota exceeded"), "rate_limited"),
    (ProviderError(503), "temporarily_unavailable"),
    (TimeoutError(), "temporarily_unavailable"),
    (ProviderError(401), "invalid"),
])
def test_error_classification(error, expected):
    assert providers.classify_error(error).status == expected


def test_model_configuration_error_skips_remaining_keys_for_that_provider(monkeypatch):
    add_system("openai", "bad-config-A")
    add_system("openai", "bad-config-B")
    add_system("groq", "working-groq")
    calls = []

    def build(config, api_key):
        model = Mock()
        calls.append(api_key)
        if config["name"] == "openai":
            model.invoke.side_effect = ProviderError(404, "model not found")
        else:
            model.invoke.return_value = "ok"
        return model

    monkeypatch.setattr(providers, "build_chat_model", build)
    assert providers.PooledChatModel("nobody").invoke("hello") == "ok"
    assert len([key for key in calls if key.startswith("bad-config")]) == 1


def test_custom_openai_compatible_provider_is_extensible():
    providers.save_provider_config("custom_ai", "openai_compatible", "https://example.test/v1",
                                   "custom-model", 4096, 25, 1, True)
    providers.save_system_provider_key("custom_ai", "custom-provider-key")
    candidate = vault.request_candidates("nobody")[0]
    assert candidate.provider == "custom_ai"
    assert candidate.config["endpoint"] == "https://example.test/v1"


def test_pool_raises_only_after_every_candidate_fails(monkeypatch):
    add_system("groq", "failure-A")
    add_system("anthropic", "failure-B")
    model = Mock()
    model.invoke.side_effect = ProviderError(503)
    monkeypatch.setattr(providers, "build_chat_model", lambda config, key: model)
    with pytest.raises(providers.ProviderPoolExhausted) as raised:
        providers.PooledChatModel("nobody").invoke("hello")
    assert len(raised.value.attempts) == 2
    assert "failure-A" not in str(raised.value)


def test_tool_binding_returns_an_independent_pool_wrapper():
    model = providers.PooledChatModel("alice")
    first = model.bind_tools(["first"])
    second = model.bind_tools(["second"])
    assert first is not second and first is not model
    assert first.tools == ["first"] and second.tools == ["second"]


def test_visible_content_excludes_reasoning_blocks():
    content = [{"type": "reasoning", "text": "private"}, {"type": "text", "text": "visible"}]
    assert providers.content_text(content) == "visible"


def test_encrypted_tavily_key_is_not_returned_by_status():
    providers.save_search_key("alice", "tvly-super-secret")
    status = providers.search_key_status("alice")
    assert status["configured"]
    assert "super-secret" not in repr(status)
    assert vault.get_service_key("alice", "tavily") == "tvly-super-secret"
