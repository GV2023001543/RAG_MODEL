from unittest.mock import Mock

import provider_adapters as adapters


BASE = {
    'name': 'test', 'model': 'test-model', 'endpoint': '',
    'max_tokens': 321, 'timeout': 17,
}


def test_custom_adapter_can_be_registered_without_scheduler_changes():
    builder = Mock(return_value='client')
    adapters.register_adapter('test_adapter')(builder)
    config = {**BASE, 'adapter': 'test_adapter'}
    assert adapters.build_chat_model(config, 'secret-key') == 'client'
    builder.assert_called_once_with(config, 'secret-key')


def test_groq_adapter_receives_endpoint_and_no_internal_retries(monkeypatch):
    constructor = Mock(return_value='groq-client')
    import langchain_groq
    monkeypatch.setattr(langchain_groq, 'ChatGroq', constructor)
    config = {**BASE, 'name': 'groq', 'adapter': 'groq', 'endpoint': 'https://groq.test/v1'}
    adapters.build_chat_model(config, 'groq-secret')
    kwargs = constructor.call_args.kwargs
    assert kwargs['groq_api_base'] == 'https://groq.test/v1'
    assert kwargs['max_retries'] == 0
    assert kwargs['max_tokens'] == 321
    assert kwargs['timeout'] == 17


def test_openai_compatible_adapter_uses_configured_endpoint(monkeypatch):
    constructor = Mock(return_value='openai-client')
    import langchain_openai
    monkeypatch.setattr(langchain_openai, 'ChatOpenAI', constructor)
    config = {**BASE, 'name': 'custom', 'adapter': 'openai_compatible', 'endpoint': 'https://custom.test/v1'}
    adapters.build_chat_model(config, 'custom-secret')
    kwargs = constructor.call_args.kwargs
    assert kwargs['base_url'] == 'https://custom.test/v1'
    assert kwargs['max_retries'] == 0


def test_gemini_adapter_uses_common_limits(monkeypatch):
    constructor = Mock(return_value='gemini-client')
    import langchain_google_genai
    monkeypatch.setattr(langchain_google_genai, 'ChatGoogleGenerativeAI', constructor)
    config = {**BASE, 'name': 'gemini', 'adapter': 'gemini', 'endpoint': 'gemini.test'}
    adapters.build_chat_model(config, 'gemini-secret')
    kwargs = constructor.call_args.kwargs
    assert kwargs['client_options'] == {'api_endpoint': 'gemini.test'}
    assert kwargs['retries'] == 0
    assert kwargs['request_timeout'] == 17


def test_anthropic_adapter_uses_endpoint_and_no_internal_retries(monkeypatch):
    constructor = Mock(return_value='anthropic-client')
    import langchain_anthropic
    monkeypatch.setattr(langchain_anthropic, 'ChatAnthropic', constructor)
    config = {
        **BASE, 'name': 'anthropic', 'adapter': 'anthropic',
        'endpoint': 'https://anthropic.test',
    }
    adapters.build_chat_model(config, 'anthropic-secret')
    kwargs = constructor.call_args.kwargs
    assert kwargs['base_url'] == 'https://anthropic.test'
    assert kwargs['max_retries'] == 0
    assert kwargs['max_tokens'] == 321
    assert kwargs['timeout'] == 17
