"""UI acceptance tests for state, filters and explicitly triggered AI."""

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

from components import consolidated_view, data_upload
from utils.data_processing import parse_csv

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def button(app, label):
    return next(b for b in app.button if b.label == label)


def upload(payload, name="history.csv"):
    return SimpleNamespace(name=name, getvalue=lambda: payload)


def test_four_tabs_empty_states_no_requests(monkeypatch):
    summary = Mock()
    monkeypatch.setattr(consolidated_view, "summarize", summary)
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Data Upload",
        "Consolidated View",
        "Historical Performance",
        "AI Analyst (Chat)",
    ]
    assert not app.metric
    summary.assert_not_called()


def test_valid_invalid_replacement_and_clear(monkeypatch):
    payload = Path("data/sample_sauna_history.csv").read_bytes()
    monkeypatch.setattr(data_upload.st, "file_uploader", lambda *a, **k: upload(payload))
    app = AppTest.from_file(APP_PATH).run()
    button(app, "Validate and use CSV").click().run()
    assert not app.exception
    assert len(app.session_state.data) == 17
    assert any(m.label == "Maximum sessions per week" and m.value == "4" for m in app.metric)
    app.session_state.chat_history = [{"role": "user", "content": "retained conversation"}]
    monkeypatch.setattr(
        data_upload.st, "file_uploader", lambda *a, **k: upload(b"date\ninvalid", "bad.csv")
    )
    app.run()
    button(app, "Validate and use CSV").click().run()
    assert not app.exception
    assert len(app.session_state.data) == 17
    assert app.session_state.active_name == "history.csv"
    assert app.session_state.chat_history[0]["content"] == "retained conversation"
    assert any("missing columns" in e.value for e in app.error)
    button(app, "Clear uploaded data").click().run()
    assert not app.exception
    assert "data" not in app.session_state
    assert not app.metric
    assert app.session_state.chat_history == []


def test_filter_cards_summary_reruns_and_invalidation(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    summary = Mock(
        return_value="You recorded six sessions. Track consistently over the next six months."
    )
    monkeypatch.setattr(consolidated_view, "summarize", summary)
    app = AppTest.from_file(APP_PATH).run()
    app.session_state.data = parse_csv(
        Path("data/sample_sauna_history.csv").read_bytes(), "Celsius", today=date(2026, 10, 10)
    ).data
    app.session_state.active_name = "history.csv"
    app.session_state.active_input_unit = "Celsius"
    app.run()
    next(c for c in app.checkbox if c.label == "I understand and enable AI requests").check().run()
    app.date_input[0].set_value((date(2026, 9, 1), date(2026, 9, 30))).run()
    assert not app.exception
    assert any(m.label == "Duration" and m.value == "15.5 min" for m in app.metric)
    summary.assert_not_called()
    button(app, "Generate summary").click().run()
    assert not app.exception
    assert summary.call_count == 1
    app.run()
    assert summary.call_count == 1 and app.session_state.summary is not None
    app.date_input[0].set_value((date(2026, 2, 1), date(2026, 2, 28))).run()
    assert not app.exception
    assert not app.metric
    assert app.session_state.summary is None
    app.date_input[0].set_value((date(2026, 9, 1), date(2026, 9, 30))).run()
    assert app.session_state.summary is None
    assert summary.call_count == 1


def test_local_chat_no_api_key(monkeypatch):
    from components import ai_analyst

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")
    answer = Mock(return_value="September 2026 had six recorded sessions.")
    monkeypatch.setattr(ai_analyst, "chat", answer)
    app = AppTest.from_file(APP_PATH).run()
    app.session_state.data = parse_csv(
        Path("data/sample_sauna_history.csv").read_bytes(), "Celsius", today=date(2026, 10, 11)
    ).data
    app.session_state.active_name = "history.csv"
    app.session_state.active_input_unit = "Celsius"
    app.run()
    assert app.chat_input[0].disabled
    next(c for c in app.checkbox if c.label == "I understand and enable AI requests").check().run()
    assert not app.chat_input[0].disabled
    app.chat_input[0].set_value("Summarize September 2026").run()
    assert not app.exception
    assert answer.call_args.args[4:6] == ("http://localhost:11434", "qwen2.5:7b")
    assert app.session_state.chat_history[-1]["content"] == answer.return_value
