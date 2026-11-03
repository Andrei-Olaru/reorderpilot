"""Streamlit UI tests: run the app headless (AppTest) against a fake API, no server/model/LLM needed."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

FAKE = {
    ("GET", "/health"): {"status": "ok", "as_of": "2011-11-01"},
    ("POST", "/agent/chat"): {
        "answer": "Customer 16121: WHITE HANGING HEART T-LIGHT HOLDER is suppressed (overstocked).",
        "tool_calls": [{"name": "predict_reorder", "args": {"customer_id": "16121"}, "result_summary": "5 items"}],
        "trace_url": None,
    },
    ("GET", "/memory/16121"): {"customer_id": "16121", "notes": [], "feedback_events": [], "price_sensitive": False},
    ("GET", "/predict/16121"): {
        "customer_id": "16121", "as_of": "2011-11-01", "abstain": False, "why": None, "price_sensitive": False,
        "recommendations": [{"stock_code": "85123A", "description": "WHITE HANGING HEART T-LIGHT HOLDER",
                             "probability": 0.42, "reason_text": "bought 6 times", "suppressed": True,
                             "suppressed_reason": "declined: overstocked until 2011-12-01"}],
    },
}


@pytest.fixture
def fake_api(monkeypatch):
    """Replace httpx.request with a router over FAKE; records every call."""
    calls: list[tuple[str, str, dict]] = []

    def request(method, url, **kwargs):
        path = "/" + url.split("/", 3)[3]
        calls.append((method, path, kwargs))
        payload = FAKE.get((method, path))
        if payload is None:
            return httpx.Response(404, json={"detail": "not found"})
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(httpx, "request", request)
    return calls


def test_app_survives_api_down(monkeypatch):
    def down(*_, **__):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "request", down)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("not reachable" in e.value for e in at.error)


def test_chat_shows_answer_and_tool_calls(fake_api):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("Brief me on customer 16121").run()
    assert not at.exception
    assert ("POST", "/agent/chat") in [(method, path) for method, path, _ in fake_api]
    assert "suppressed" in at.chat_message[1].markdown[0].value
    assert at.expander[0].label == "1. predict_reorder"


def test_memory_tab_shows_suppressed_prediction(fake_api):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["customer_id"] = "16121"
    at.run()
    assert not at.exception
    rows = at.dataframe[-1].value
    assert rows.loc[0, "stock_code"] == "85123A"
    assert rows.loc[0, "suppressed"].startswith("yes")

