"""Live-call copilot endpoints for the /live page (SPEC section 17, V4b; DECISIONS D31-D33).

    browser (2 audio streams, voice detection) --utterance WAV--> POST /copilot/transcribe --> "[CLIENT] ..." line
    browser (transcript so far)                --JSON-->          POST /copilot/turn       --> battle cards
    browser (note + declined products)         --JSON-->          POST /copilot/end_call   --> memory.db (+ index check)

Every endpoint is a thin wrapper over copilot/call_session.py, so the Streamlit tab and this API share one
implementation. The backend (LLM, data, Chroma) is built once per process; tests override `get_copilot_backend`.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from reorderpilot.agent.llm import LLMError
from reorderpilot.api.schemas import (
    CopilotTurnRequest,
    CopilotTurnResponse,
    EndCallRequest,
    EndCallResponse,
    ProductOption,
    Speaker,
    TranscribeResponse,
)
from reorderpilot.copilot import call_session
from reorderpilot.copilot.audio_engine import DEMO_CALL, AudioFileError, TranscriptionError, max_upload_bytes

router = APIRouter(prefix="/copilot", tags=["copilot"])
log = logging.getLogger(__name__)

# Content-Type of the request body -> file suffix the audio engine (and Groq) understands.
CONTENT_TYPES = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav", "audio/webm": ".webm",
                 "audio/ogg": ".ogg", "audio/mpeg": ".mp3", "audio/mp4": ".mp4", "audio/flac": ".flac"}


@lru_cache(maxsize=1)
def _real_backend() -> call_session.CopilotBackend:
    """Built on the first copilot request, then reused (model, data and Chroma load once per process)."""
    return call_session.default_backend()


def get_copilot_backend() -> call_session.CopilotBackend:
    """Dependency: the real backend (tests override it with fakes)."""
    return _real_backend()


Backend = Annotated[call_session.CopilotBackend, Depends(get_copilot_backend)]


def _http_error(exc: Exception) -> HTTPException:
    """Known errors -> a status code the page can act on, with the same readable message the Streamlit tab shows."""
    if isinstance(exc, TranscriptionError):
        status = 502   # upstream (Groq) failed
    elif isinstance(exc, LLMError):
        status = 503   # LLM unavailable; retry later
    else:
        status = 400   # AudioFileError / ValueError: the request itself is wrong
    return HTTPException(status_code=status, detail=call_session.describe_error(exc))


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


@router.get("/demo")
def demo_call() -> dict[str, list[str]]:
    """The scripted demo call (same lines as the CLI and the Streamlit tab)."""
    return {"lines": list(DEMO_CALL)}


@router.post("/transcribe")
async def transcribe(request: Request, backend: Backend,
                     speaker: Annotated[Speaker | Literal["AUTO"], Query()],
                     previous: Annotated[Speaker | None, Query()] = None) -> TranscribeResponse:
    """One utterance (raw audio body) -> one transcript line, or null if nothing was said.

    speaker=CLIENT/SALES_REP when the stream tells who speaks (two-stream mode); AUTO with one microphone,
    then the line is labelled from cue words and the `previous` speaker (D26 rules).
    """
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    suffix = CONTENT_TYPES.get(content_type)
    if suffix is None:
        raise HTTPException(415, f"unsupported Content-Type '{content_type}'; use one of {sorted(CONTENT_TYPES)}")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_upload_bytes():
        raise HTTPException(413, f"audio is {int(declared):,} bytes, over the {max_upload_bytes():,}-byte limit")
    data = await request.body()
    start = time.perf_counter()
    try:
        line = await run_in_threadpool(call_session.transcribe_utterance, data, suffix, speaker, backend, previous)
    except (AudioFileError, TranscriptionError, ValueError) as exc:
        raise _http_error(exc) from exc
    return TranscribeResponse(line=line, latency_ms=_ms(start))


@router.post("/turn")
def turn(req: CopilotTurnRequest, backend: Backend) -> CopilotTurnResponse:
    """Run the LangGraph copilot on the transcript so far and return the battle cards."""
    start = time.perf_counter()
    try:
        state = call_session.run_turn(backend, req.customer_id, "\n".join(req.transcript))
    except (LLMError, ValueError) as exc:
        raise _http_error(exc) from exc
    prediction = state.get("prediction") or {}
    return CopilotTurnResponse(
        cards=state.get("cards", []), trail=state.get("trail", []), stop_reason=state.get("stop_reason", ""),
        removed_codes=state.get("removed_codes", []), memories=state.get("memories", []),
        declinable=[ProductOption(stock_code=c, description=d)
                    for c, d in call_session.declinable_products(prediction).items()],
        price_sensitive=bool(prediction.get("price_sensitive")),
        suggested_note=call_session.default_note(req.transcript, backend.ctx.as_of.isoformat()),
        latency_ms=_ms(start))


@router.post("/end_call")
def end_call(req: EndCallRequest, backend: Backend) -> EndCallResponse:
    """Save the call note + declined products through memory/store.py, then check the vector index (D24, D30)."""
    try:
        saved = call_session.persist_call(backend.ctx, req.customer_id, req.note, req.declined, req.reason)
    except ValueError as exc:
        raise _http_error(exc) from exc
    try:
        check = call_session.check_vector_memory(backend.memory, backend.ctx, saved.note, req.transcript)
        error = None
    except Exception as exc:  # the note IS saved in SQLite; only the index check failed, so report it, don't 500
        log.exception("vector memory check failed")
        check, error = call_session.VectorCheck(indexed=False), call_session.describe_error(exc)
    return EndCallResponse(note=saved.note, effects=saved.effects, vector_indexed=check.indexed,
                           vector_rank=check.rank, vector_distance=check.distance, vector_error=error)
