"""/copilot/* endpoints and the /live page (SPEC section 17, V4b): fake backend, fake Whisper, no network."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from reorderpilot.api.copilot import get_copilot_backend
from reorderpilot.api.main import app
from reorderpilot.copilot import audio_engine
from reorderpilot.copilot.audio_engine import Segment, TranscriptionError
from tests.test_agent_loop import ScriptedLLM, answer

CALL = ["[SALES_REP] Hi, anything for the next order?",
        "[CLIENT] Your last delivery was late and the lunch bags sell well."]


def cards(*titles: str) -> dict:
    return answer(json.dumps({"cards": [{"title": t, "say": "Lunch bags sell well for you.", "evidence": "data"}
                                        for t in titles]}))


@pytest.fixture
def client(backend):
    app.dependency_overrides[get_copilot_backend] = lambda: backend
    yield TestClient(app)
    app.dependency_overrides.clear()


def fake_whisper(monkeypatch, *texts: str) -> None:
    """Replace the Groq call: return these texts as segments, ignoring the audio bytes."""
    segments = [Segment(float(i), float(i) + 1, t) for i, t in enumerate(texts)]
    monkeypatch.setattr(audio_engine, "transcribe", lambda *_args, **_kwargs: segments)


def post_audio(client, body: bytes = b"RIFF fake", content_type: str = "audio/wav", **params):
    return client.post("/copilot/transcribe", params=params, content=body, headers={"Content-Type": content_type})


def test_turn_returns_cards_route_products_and_note(client, backend):
    backend.llm = ScriptedLLM([cards("Offer the lunch bags")])
    res = client.post("/copilot/turn", json={"customer_id": "A", "transcript": CALL})
    assert res.status_code == 200
    body = res.json()
    assert [c["title"] for c in body["cards"]] == ["Offer the lunch bags"]
    assert body["trail"] == ["supervisor", "ml_prediction", "vector_memory", "live_copilot", "grounding_guard"]
    assert "P1" in [p["stock_code"] for p in body["declinable"]]
    assert "delivery was late" in body["suggested_note"]
    assert "anything for the next order" not in body["suggested_note"]      # the note quotes the client only
    assert body["latency_ms"] >= 0


def test_turn_rejects_lines_without_a_speaker_tag(client):
    res = client.post("/copilot/turn", json={"customer_id": "A", "transcript": ["hello there"]})
    assert res.status_code == 422


def test_transcribe_labels_the_line_with_the_stream_speaker(client, monkeypatch):
    fake_whisper(monkeypatch, " Your prices", "went up.")
    res = post_audio(client, speaker="CLIENT")
    assert res.status_code == 200
    assert res.json()["line"] == "[CLIENT] Your prices went up."


def test_transcribe_drops_whisper_silence_hallucination(client, monkeypatch):
    fake_whisper(monkeypatch, " Thank you.")
    assert post_audio(client, speaker="CLIENT").json()["line"] is None


@pytest.mark.parametrize("text, previous, expected", [
    ("Your prices went up again.", "CLIENT", "[CLIENT]"),     # cue words beat turn-taking
    ("Sounds good to me.", "CLIENT", "[SALES_REP]"),          # no cue: the other person than last time
])
def test_transcribe_auto_guesses_speaker_with_one_microphone(client, monkeypatch, text, previous, expected):
    fake_whisper(monkeypatch, text)
    line = post_audio(client, speaker="AUTO", previous=previous).json()["line"]
    assert line.startswith(expected)


def test_transcribe_errors_are_readable_status_codes(client, monkeypatch):
    assert post_audio(client, content_type="text/plain", speaker="CLIENT").status_code == 415
    empty = post_audio(client, body=b"", speaker="CLIENT")
    assert empty.status_code == 400 and empty.json()["detail"].startswith("Audio file problem")

    def down(*_args, **_kwargs):
        raise TranscriptionError("gave up after 3 attempts: APITimeoutError")

    monkeypatch.setattr(audio_engine, "transcribe", down)
    failed = post_audio(client, speaker="CLIENT")
    assert failed.status_code == 502 and failed.json()["detail"].startswith("Transcription failed")


def test_end_call_persists_and_the_next_turn_retrieves_the_note(client, backend):
    backend.llm = ScriptedLLM([cards("Card")])
    note = "Live call. Client said: Your last delivery was late and the lunch bags sell well."
    saved = client.post("/copilot/end_call", json={"customer_id": "A", "transcript": CALL, "note": note,
                                                   "declined": ["P1"], "reason": "overstocked"})
    assert saved.status_code == 200
    body = saved.json()
    assert body["effects"] == ["P1 suppressed until 2011-03-31"]
    assert body["vector_indexed"] and body["vector_rank"] == 1

    turn = client.post("/copilot/turn", json={"customer_id": "A", "transcript": CALL}).json()
    assert note in [m["text"] for m in turn["memories"]]
    assert "P1" not in [p["stock_code"] for p in turn["declinable"]]      # declined -> suppressed next call


def test_end_call_rejects_an_empty_note(client):
    res = client.post("/copilot/end_call", json={"customer_id": "A", "note": ""})
    assert res.status_code == 422


def test_live_page_and_its_scripts_are_served(client):
    page = client.get("/live/")
    assert page.status_code == 200 and "<title>ReorderPilot Live</title>" in page.text
    assert client.get("/live/live.js").status_code == 200
    assert client.get("/live/frame-tap.js").status_code == 200
