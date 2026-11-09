"""Live-call session helpers for the Streamlit "Live Call Copilot" tab (SPEC section 17, milestone V4; D28-D30).

The tab stays a thin layer of widgets; everything it *does* lives here as plain functions so it can be unit-tested
without Streamlit:

    lines (simulated / uploaded audio / pasted) --live_session--> run_graph after each CLIENT line --> cards
    "End call & persist" --> memory/store.py (note + declined feedback) --> check the note is in vector memory

SQLite stays the source of truth (D24): notes are written through store.py only; Chroma is re-synced from it.
"""

from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

from reorderpilot.agent import graph, tools
from reorderpilot.agent.key_rotator import GroqKeyRotator
from reorderpilot.agent.llm import ChatModel, LLMError
from reorderpilot.api.schemas import CustomerNote, FeedbackEvent, Reason
from reorderpilot.copilot import audio_engine
from reorderpilot.memory import store
from reorderpilot.memory.vector_store import VectorMemory

logger = logging.getLogger(__name__)
ELLIPSIS = "…"


@dataclass
class CopilotBackend:
    """Everything a live call needs. The app builds the real one; tests pass fakes (scripted LLM, temp DB)."""

    llm: ChatModel
    ctx: tools.ToolContext
    memory: VectorMemory | None
    make_rotator: Callable[[], GroqKeyRotator] = field(default=audio_engine.make_rotator)
    tracing: bool = False


def default_backend() -> CopilotBackend:
    """Real data, memory.db, Chroma index and chat LLM (same wiring as audio_engine._graph_runner)."""
    from reorderpilot.agent.llm import LLMConfig, OpenAICompatibleLLM
    from reorderpilot.config import get_settings

    s = get_settings()
    return CopilotBackend(llm=OpenAICompatibleLLM(LLMConfig.from_env()),
                          ctx=tools.ToolContext(as_of=s.as_of_date, db_path=s.memory_db_path),
                          memory=VectorMemory(s.chroma_dir), tracing=s.langsmith_tracing)


def describe_error(exc: Exception) -> str:
    """One readable line for the UI (no traceback): known errors get a plain label, others name their type."""
    if isinstance(exc, audio_engine.AudioFileError):
        return f"Audio file problem: {exc}"
    if isinstance(exc, audio_engine.TranscriptionError):
        return f"Transcription failed: {exc}"
    if isinstance(exc, LLMError):
        return f"LLM unavailable: {exc}"
    if isinstance(exc, ValueError):
        return f"Invalid input: {exc}"
    return f"Unexpected error ({type(exc).__name__}): {exc}. Details are in the terminal log."


def run_turn(backend: CopilotBackend, customer_id: str, transcript: str) -> dict:
    """One copilot turn: the whole transcript so far -> final graph state (cards, trail, memories, ...)."""
    return dict(graph.run_graph(transcript, backend.llm, backend.ctx, backend.memory, customer_id=customer_id,
                                tracing=backend.tracing))


def make_runner(backend: CopilotBackend, customer_id: str) -> Callable[[str], dict]:
    """transcript text -> final graph state, or {"error": "..."} so one failed turn does not end the call."""

    def run(text: str) -> dict:
        try:
            return run_turn(backend, customer_id, text)
        except Exception as exc:  # UI boundary: show the error in the tab, keep the full traceback in the log
            logger.exception("copilot turn failed")
            return {"error": describe_error(exc)}

    return run


@contextmanager
def _temp_audio(data: bytes, suffix: str) -> Iterator[Path]:
    """Write audio bytes to a validated temp file (the audio engine reads files); always deleted afterwards."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"call{suffix.lower()}"
        path.write_bytes(data)
        audio_engine.validate_audio_file(path)  # fail fast: size/format checked before any key or network setup
        yield path


def transcribe_upload(filename: str, data: bytes, backend: CopilotBackend,
                      llm_labels: bool = False) -> tuple[list[str], str]:
    """Uploaded bytes -> ('[CLIENT] ...' lines, diarization method). The temp copy is always deleted."""
    with _temp_audio(data, Path(filename).suffix) as path:
        return audio_engine.transcribe_call(path, backend.make_rotator(), llm=backend.llm if llm_labels else None)


# Whisper "hears" these on near-silence or noise. Dropped only when they are the WHOLE utterance (D32).
SILENCE_HALLUCINATIONS = frozenset({"thank you", "thanks for watching", "thank you for watching", "you", "bye",
                                    "subtitles by the amaraorg community", "so"})


def is_silence_hallucination(text: str) -> bool:
    """True if the whole utterance is a phrase Whisper typically invents on silence."""
    return re.sub(r"[^a-z ]", "", text.lower()).strip() in SILENCE_HALLUCINATIONS


AUTO = "AUTO"


def guess_speaker(text: str, previous: str | None) -> str:
    """Microphone-only mode: cue words decide (D26 rules); otherwise the other person than last time speaks."""
    return audio_engine.cue_speaker(text) or (audio_engine.other(previous) if previous else audio_engine.SALES_REP)


def transcribe_utterance(data: bytes, suffix: str, speaker: str, backend: CopilotBackend,
                         previous: str | None = None) -> str | None:
    """One utterance -> '[SPEAKER] text', or None if nothing was said.

    `speaker` is CLIENT or SALES_REP when it is known from the audio stream (two-stream mode, exact), or AUTO
    with only one microphone: then it is guessed from the text and the `previous` speaker.
    """
    if speaker not in (*audio_engine.SPEAKERS, AUTO):
        raise ValueError(f"speaker must be one of {(*audio_engine.SPEAKERS, AUTO)}")
    with _temp_audio(data, suffix) as path:
        segments = audio_engine.transcribe(path, backend.make_rotator())
    text = " ".join(audio_engine.clean_text(s.text) for s in segments).strip()
    if not text or is_silence_hallucination(text):
        return None
    label = guess_speaker(text, previous) if speaker == AUTO else speaker
    return audio_engine.to_transcript_lines([replace(segments[0], text=text, speaker=label)])[0]


def declinable_products(prediction: dict | None) -> dict[str, str]:
    """stock_code -> description of the products the copilot could offer (unsuppressed recommendations)."""
    return {r["stock_code"]: r["description"] for r in (prediction or {}).get("recommendations", [])
            if not r.get("suppressed")}


def default_note(lines: list[str], as_of: str) -> str:
    """Pre-filled call note: what the client said, verbatim (no LLM, so nothing invented), max 500 chars."""
    said = [line.removeprefix("[CLIENT] ") for line in lines if line.startswith("[CLIENT]")]
    note = f"Live call {as_of}. Client said: " + " | ".join(said) if said else f"Live call {as_of}."
    if len(note) > store.NOTE_MAX_CHARS:
        note = note[: store.NOTE_MAX_CHARS - len(ELLIPSIS)] + ELLIPSIS
    return note


@dataclass(frozen=True)
class SavedCall:
    """What "End call & persist" wrote: the note row and the policy effect of each declined product."""

    note: CustomerNote
    effects: list[str]


def persist_call(ctx: tools.ToolContext, customer_id: str, note: str, declined: list[str],
                 reason: Reason) -> SavedCall:
    """Save the call note, then one 'declined' feedback event per product (the v1 suppression policy applies)."""
    saved = store.add_customer_note(ctx.db_path, customer_id, note.strip(), ctx.as_of)  # validates before feedback
    effects = []
    for code in declined:
        event = FeedbackEvent(customer_id=customer_id, outcome="declined", stock_code=code, reason=reason)
        effects.append(store.record_feedback(ctx.db_path, event, ctx.as_of).effect)
    return SavedCall(note=saved, effects=effects)


@dataclass(frozen=True)
class VectorCheck:
    """Is the saved note in the Chroma index, and would the copilot retrieve it for this call's client lines?"""

    indexed: bool
    rank: int | None = None        # 1-based position in the top-k search results, None if not in the top k
    distance: float | None = None


def check_vector_memory(memory: VectorMemory | None, ctx: tools.ToolContext, note: CustomerNote,
                        transcript: list[str]) -> VectorCheck:
    """Sync SQLite -> Chroma like the next call will, then search with the same query the copilot uses."""
    if memory is None:
        return VectorCheck(indexed=False)
    memory.sync_customer(ctx.db_path, note.customer_id, ctx.as_of)
    indexed = bool(memory.collection.get(ids=[f"note-{note.id}"])["ids"])
    hits = memory.search(note.customer_id, graph.memory_query(transcript), ctx.as_of, k=graph.MEMORY_K)
    for rank, hit in enumerate(hits, start=1):
        if hit.source == "note" and hit.text == note.note:
            return VectorCheck(indexed=indexed, rank=rank, distance=hit.distance)
    return VectorCheck(indexed=indexed)
