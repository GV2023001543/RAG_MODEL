from streamlit.testing.v1 import AppTest


def test_workspace_and_authenticated_settings_render(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "test-admin-password")
    app = AppTest.from_file("langgraph_rag_frontend.py")
    app.session_state["admin_authenticated"] = True

    app.run(timeout=30)
    assert not app.exception
    assert any(button.label == "New chat" for button in app.button)
    assert any(field.key == "page_history_search" for field in app.text_input)
    settings_buttons = [button for button in app.button if button.label == "Settings"]
    assert len(settings_buttons) == 2

    next(button for button in settings_buttons if button.key == "header_settings").click().run(timeout=30)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Personal AI keys", "Appearance & data", "Admin pool"
    ]
    assert {select.label for select in app.selectbox} >= {
        "Provider", "Configure provider", "Adapter", "System key provider"
    }


def test_admin_password_form_renders_when_server_secret_is_configured(monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "test-admin-password")
    app = AppTest.from_file("langgraph_rag_frontend.py").run(timeout=30)
    next(button for button in app.button if button.key == "header_settings").click().run(timeout=30)

    assert not app.exception
    assert any(field.label == "Admin password" for field in app.text_input)
    assert any(button.label == "Unlock admin pool" for button in app.button)
