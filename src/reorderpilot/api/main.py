"""FastAPI app exposing the SPEC section 9 endpoints.

Heavy collaborators (model, agent) are injected with FastAPI dependencies (`get_predictor`, `get_due_lister`,
`get_chat_handler`, `get_db_path`, `get_as_of`). Tests replace them via `app.dependency_overrides`, so the API can
be tested without a trained model, an LLM or the real memory.db.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Query, Request
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from reorderpilot.agent.loop import make_chat_handler
from reorderpilot.api.copilot import router as copilot_router
from reorderpilot.api.schemas import (
    ChatRequest,
    ChatResponse,
    CustomersDueResult,
    FeedbackEvent,
    MemoryResult,
    PredictionResult,
    WriteMemoryResult,
)
from reorderpilot.config import LIVE_UI_DIR, Settings, get_settings
from reorderpilot.memory import store

Predictor = Callable[[str, int, date], PredictionResult]  # (customer_id, top_k, as_of) -> raw model output
DueLister = Callable[[int, date, Path], CustomersDueResult]  # (limit, as_of, db_path)
ChatHandler = Callable[[ChatRequest], ChatResponse]

app = FastAPI(title="ReorderPilot", version="0.1.0")
log = logging.getLogger(__name__)
app.include_router(copilot_router)  # /copilot/* (SPEC section 17, V4b)
app.mount("/live", StaticFiles(directory=LIVE_UI_DIR, html=True), name="live")  # the live-call page


# ---------- error handlers: readable JSON instead of a bare 500 ----------

@app.exception_handler(sqlite3.OperationalError)
def _db_unavailable(_: Request, exc: sqlite3.OperationalError) -> JSONResponse:
    """Memory DB locked or unreadable -> 503 so clients know to retry."""
    log.warning("memory db error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "Memory database is busy or unavailable; retry shortly."})


@app.exception_handler(FileNotFoundError)
def _model_missing(_: Request, exc: FileNotFoundError) -> JSONResponse:
    """Model artifact not trained yet -> 503 with the fix in the message."""
    log.warning("missing file: %s", exc)
    return JSONResponse(status_code=503, content={"detail": f"Model not available: {exc}"})


# ---------- dependencies (overridable in tests) ----------

def get_app_settings() -> Settings:
    """Settings from the environment (OS env first, .env fallback)."""
    return get_settings()


def get_as_of(settings: Annotated[Settings, Depends(get_app_settings)]) -> date:
    """Simulated 'today' (AS_OF_DATE), never the wall clock."""
    return settings.as_of_date


def get_db_path(settings: Annotated[Settings, Depends(get_app_settings)]) -> Path:
    """Path of the SQLite memory database."""
    return settings.memory_db_path


def get_predictor() -> Predictor:
    """The real model's predict_reorder (imported lazily so the API starts even if the model track is unfinished)."""
    from reorderpilot.model.predict import predict_reorder

    return lambda customer_id, top_k, as_of: predict_reorder(customer_id, top_k=top_k, as_of=as_of)


def get_due_lister() -> DueLister:
    """The agent tool list_customers_due, adapted to plain arguments."""
    from reorderpilot.agent.tools import ListCustomersDueArgs, ToolContext, list_customers_due

    return lambda limit, as_of, db_path: list_customers_due(
        ListCustomersDueArgs(limit=limit), ToolContext(as_of=as_of, db_path=db_path)
    )


def get_chat_handler() -> ChatHandler:
    """The agent loop entry point (real LLM + real memory.db; tests override this dependency)."""
    return make_chat_handler()


AsOf = Annotated[date, Depends(get_as_of)]
DbPath = Annotated[Path, Depends(get_db_path)]
CustomerIdPath = Annotated[str, PathParam(min_length=1, max_length=20, pattern=r"^[0-9A-Za-z]+$")]


# ---------- endpoints ----------

@app.get("/health")
def health(as_of: AsOf) -> dict:
    """Liveness check; also shows the simulated date the server is running at."""
    return {"status": "ok", "as_of": as_of.isoformat()}


@app.get("/customers/due")
def customers_due(
    as_of: AsOf, db_path: DbPath, lister: Annotated[DueLister, Depends(get_due_lister)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> CustomersDueResult:
    """Customers ranked by expected re-order value."""
    return lister(limit, as_of, db_path)


@app.get("/predict/{customer_id}")
def predict(
    customer_id: CustomerIdPath, as_of: AsOf, db_path: DbPath,
    predictor: Annotated[Predictor, Depends(get_predictor)],
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
) -> PredictionResult:
    """Model predictions with memory suppression and the price-sensitive flag applied (same policy as the agent)."""
    raw = predictor(customer_id, top_k, as_of)
    return store.apply_suppressions(raw, db_path, as_of)


@app.post("/feedback", status_code=201)
def feedback(event: FeedbackEvent, as_of: AsOf, db_path: DbPath) -> WriteMemoryResult:
    """Record a feedback event; the response says what the policy did (e.g. 'suppressed until ...')."""
    return store.record_feedback(db_path, event, as_of)


@app.get("/memory/{customer_id}")
def memory(customer_id: CustomerIdPath, as_of: AsOf, db_path: DbPath) -> MemoryResult:
    """Notes, non-expired feedback events and price-sensitive flag for a customer."""
    return store.read_memory(db_path, customer_id, as_of)


@app.post("/agent/chat")
def agent_chat(request: ChatRequest, handler: Annotated[ChatHandler, Depends(get_chat_handler)]) -> ChatResponse:
    """One chat turn with the sales agent."""
    return handler(request)
