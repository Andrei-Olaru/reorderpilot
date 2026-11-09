"""Streamlit UI tests: run the app headless (AppTest) against a fake API, no server/model/LLM needed.

The Live Call Copilot tab runs in-process (D28), so its tests inject a fake CopilotBackend: scripted LLM,
tiny dataset + fake model, temp memory.db and a Chroma index with an offline word-hash embedding
(the `backend` fixture in conftest.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from reorderpilot.agent.llm import LLMError
from reorderpilot.copilot import audio_engine, call_session
from reorderpilot.copilot.audio_engine import DEMO_CALL, TranscriptionError
from reorderpilot.memory import store
from tests.test_agent_loop import AS_OF, ScriptedLLM, answer

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
CUSTOMER = "A"   # in make_tx(): buys P1 (LUNCH BAG RED RETROSPOT) regularly

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
    tools = at.expander[0]
    assert tools.label == "Tool executions (1)"
    assert "predict_reorder" in tools.markdown[0].value
    assert '"customer_id": "16121"' in tools.code[0].value


def test_memory_tab_shows_suppressed_prediction(fake_api):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["customer_id"] = "16121"
    at.run()
    assert not at.exception
    rows = at.dataframe[-1].value
    assert rows.loc[0, "stock_code"] == "85123A"
    assert rows.loc[0, "status"] == "Suppressed"
    assert rows.loc[0, "suppression_reason"].startswith("declined")


def test_kpi_cards_show_as_of_and_suppressed_count(fake_api):
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["customer_id"] = "16121"
    at.run()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["As-of date"] == "2011-11-01"
    assert metrics["Suppressed items"] == "1"



# ---------- Live Call Copilot tab (V4) ----------

def cards_reply(title: str) -> dict:
    return answer(json.dumps({"cards": [{"title": title, "say": "Lunch bags sell well for you.",
                                         "evidence": "model prediction"}]}))


def copilot_app(backend) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["customer_id"] = CUSTOMER
    at.session_state["copilot_backend"] = backend
    return at.run()


def markdown_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_simulated_call_reruns_copilot_only_after_client_lines(fake_api, backend):
    backend.llm = ScriptedLLM([cards_reply("Offer the lunch bags")])
    at = copilot_app(backend)
    at.button(key="start_simulated").click().run()
    at.button(key="call_next").click().run()             # [SALES_REP] line: no copilot run
    assert backend.llm.seen == []
    assert any("Cards appear after" in c.value for c in at.caption)
    at.button(key="call_next").click().run()             # [CLIENT] line: graph runs once
    assert not at.exception
    assert len(backend.llm.seen) == 1
    assert "Offer the lunch bags" in markdown_text(at)
    assert any("supervisor → ml_prediction → vector_memory → live_copilot → grounding_guard" in c.value
               for c in at.caption)


def test_end_call_persists_note_and_decline_and_next_call_finds_note(fake_api, backend):
    backend.llm = ScriptedLLM([cards_reply(f"Card {i}") for i in range(4)])
    at = copilot_app(backend)
    at.button(key="start_simulated").click().run()
    at.button(key="call_play").click().run()
    assert len(backend.llm.seen) == 3                    # 3 CLIENT lines in the 6-line demo call
    n = len(DEMO_CALL)
    at.multiselect(key=f"call_declined_{n}").set_value(["P1"])
    at.selectbox(key=f"call_reason_{n}").set_value("overstocked")
    next(b for b in at.button if b.label == "End call & persist").click().run()
    assert not at.exception

    db = backend.ctx.db_path
    notes = store.list_customer_notes(db, CUSTOMER, AS_OF)
    assert len(notes) == 1 and "last delivery was late" in notes[0].note
    assert "P1" in store.active_suppressions(db, CUSTOMER, AS_OF)      # v1 policy applied to the decline
    assert any("ranks 1 of" in s.value for s in at.success)           # vector check shown to the rep

    at.button(key="start_simulated").click().run()                    # next call, same customer
    at.button(key="call_next").click().run()
    at.button(key="call_next").click().run()
    memories = at.session_state["call"]["state"]["memories"]
    assert notes[0].note in [m["text"] for m in memories]


def test_paste_transcript_skips_untagged_lines(fake_api, backend):
    at = copilot_app(backend)
    at.segmented_control(key="call_source").set_value("Paste transcript").run()
    at.text_area(key="call_paste").set_value("[SALES_REP] Hi there\nsome noise\n[CLIENT] Too expensive.")
    at.button(key="start_paste").click().run()
    assert not at.exception
    assert at.session_state["call"]["total"] == 2
    assert any("1 line(s) without" in w.value for w in at.warning)


@pytest.mark.parametrize("data, fail_with, expected", [
    (b"", None, "Audio file problem"),                                   # real check: empty file, nothing sent
    (b"RIFF fake", TranscriptionError("gave up after 3 attempts"), "Transcription failed"),
    (b"RIFF fake", LLMError("401 invalid key"), "LLM unavailable"),
])
def test_upload_errors_show_st_error_not_traceback(fake_api, backend, monkeypatch, data, fail_with, expected):
    if fail_with is not None:
        def fail(*_args, **_kwargs):
            raise fail_with
        monkeypatch.setattr(audio_engine, "transcribe_call", fail)
    at = copilot_app(backend)
    at.segmented_control(key="call_source").set_value("Upload audio").run()
    at.file_uploader(key="call_audio").set_value(("call.wav", data, "audio/wav")).run()
    at.button(key="start_upload").click().run()
    assert not at.exception
    assert any(e.value.startswith(expected) for e in at.error)
    assert at.session_state["call"] is None


def test_llm_down_gives_fallback_cards_from_the_model(fake_api, backend):
    backend.llm = ScriptedLLM([LLMError("429 rate limited")])
    at = copilot_app(backend)
    at.button(key="start_simulated").click().run()
    at.button(key="call_next").click().run()
    at.button(key="call_next").click().run()
    assert not at.exception
    assert any("model prediction" in i.value for i in at.info)
    assert "(P1)" in markdown_text(at)
    assert any("fallback_llm_error" in c.value for c in at.caption)


def test_failed_copilot_turn_shows_st_error_and_call_continues(fake_api, backend, monkeypatch):
    def down(*_args, **_kwargs):
        raise LLMError("all keys cooling down")
    monkeypatch.setattr("reorderpilot.agent.graph.run_graph", down)
    at = copilot_app(backend)
    at.button(key="start_simulated").click().run()
    at.button(key="call_play").click().run()
    assert not at.exception
    assert any(e.value.startswith("LLM unavailable") for e in at.error)
    assert len(at.session_state["call"]["transcript"]) == len(DEMO_CALL)


def test_transcribe_upload_deletes_temp_file_even_on_error(backend, monkeypatch):
    seen: list[Path] = []

    def fail(path, rotator, llm=None):
        seen.append(Path(path))
        raise TranscriptionError("boom")

    monkeypatch.setattr(audio_engine, "transcribe_call", fail)
    with pytest.raises(TranscriptionError):
        call_session.transcribe_upload("my call.MP3", b"abc", backend)
    assert seen[0].suffix == ".mp3" and not seen[0].exists()


def test_default_note_quotes_the_client_only_and_fits_the_store_limit():
    note = call_session.default_note(list(DEMO_CALL), "2011-03-01")
    assert note.startswith("Live call 2011-03-01. Client said: Sure, but your last delivery")
    assert "Sam from the wholesaler" not in note
    long = call_session.default_note(["[CLIENT] " + "x" * 600], "2011-03-01")
    assert len(long) == store.NOTE_MAX_CHARS and long.endswith("…")


def test_persist_call_rejects_empty_note_before_writing_feedback(backend):
    with pytest.raises(ValueError):
        call_session.persist_call(backend.ctx, CUSTOMER, "   ", ["P1"], "overstocked")
    assert store.list_feedback_events(backend.ctx.db_path, CUSTOMER, AS_OF) == []
