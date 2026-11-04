"""SQLite memory store: feedback events, customer notes, chat messages, and the deterministic policy.

Every function takes `db_path` explicitly so tests can use a temporary file (never the real memory.db).
Dates are simulated (`as_of`), never wall-clock. Feedback events are append-only (audit trail).

Visibility rule: a row written at simulated day D is visible to lookups with as_of >= D (inclusive), so feedback
logged "today" affects today's briefing. Suppression is active while as_of < expires_at (expires_at is exclusive).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import date, timedelta
from pathlib import Path

from reorderpilot.api.schemas import (
    CustomerNote,
    FeedbackEvent,
    FeedbackEventRecord,
    MemoryResult,
    MessageRecord,
    PredictionResult,
    Reason,
    WriteMemoryResult,
)
from reorderpilot.config import SUPPRESSION_DAYS

NOTE_MAX_CHARS = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    stock_code  TEXT NULL,
    outcome     TEXT NOT NULL CHECK (outcome IN ('ordered', 'declined', 'asked_alternative')),
    reason      TEXT NULL CHECK (reason IN ('overstocked', 'price', 'seasonal', 'switched_supplier', 'other')),
    note        TEXT NULL CHECK (note IS NULL OR length(note) <= 500),
    created_at  TEXT NOT NULL,
    expires_at  TEXT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_customer ON feedback_events(customer_id);
CREATE TABLE IF NOT EXISTS customer_notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    note        TEXT NOT NULL CHECK (length(note) <= 500),
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_customer ON customer_notes(customer_id);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    tool_calls_json TEXT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a connection (creating tables if needed), commit on success, always close."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # timeout: wait up to 10s for another writer to release a lock instead of failing with "database is locked"
    with closing(sqlite3.connect(db_path, timeout=10.0)) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        with conn:  # transaction: commit or roll back
            yield conn


def init_db(db_path: Path) -> None:
    """Create the three tables if they do not exist."""
    with _connect(db_path):
        pass


def _to_date(value: str | None) -> date | None:
    """Parse an ISO date column (or NULL)."""
    return date.fromisoformat(value) if value else None


def _event_from_row(row: sqlite3.Row) -> FeedbackEventRecord:
    """Convert a feedback_events row into its Pydantic record."""
    return FeedbackEventRecord(
        id=row["id"], customer_id=row["customer_id"], stock_code=row["stock_code"], outcome=row["outcome"],
        reason=row["reason"], note=row["note"], created_at=_to_date(row["created_at"]),
        expires_at=_to_date(row["expires_at"]),
    )


# ---------- feedback_events ----------

def add_feedback_event(db_path: Path, event: FeedbackEvent, as_of: date) -> FeedbackEventRecord:
    """Store a feedback event with created_at=as_of and expires_at from the policy.

    Idempotent: an identical event (same fields, same simulated day) returns the existing row instead of a new one,
    so a double-submitted form does not create duplicate rows.
    """
    expires_at = compute_expires_at(event.outcome, event.reason, as_of)
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT * FROM feedback_events WHERE customer_id = ? AND stock_code IS ? AND outcome = ? "
            "AND reason IS ? AND note IS ? AND created_at = ? ORDER BY id LIMIT 1",
            (event.customer_id, event.stock_code, event.outcome, event.reason, event.note, as_of.isoformat()),
        ).fetchone()
        if existing is not None:
            return _event_from_row(existing)
        cur = conn.execute(
            "INSERT INTO feedback_events (customer_id, stock_code, outcome, reason, note, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (event.customer_id, event.stock_code, event.outcome, event.reason, event.note, as_of.isoformat(),
             expires_at.isoformat() if expires_at else None),
        )
        new_id = cur.lastrowid
    return FeedbackEventRecord(**event.model_dump(), id=new_id, created_at=as_of, expires_at=expires_at)


def list_feedback_events(
    db_path: Path, customer_id: str, as_of: date, include_expired: bool = False
) -> list[FeedbackEventRecord]:
    """Events for a customer created on or before as_of, oldest first, optionally including expired ones.

    Events without expires_at (e.g. ordered, declined+price) never expire.
    """
    sql = "SELECT * FROM feedback_events WHERE customer_id = ? AND created_at <= ?"
    params: list[str] = [customer_id, as_of.isoformat()]
    if not include_expired:
        sql += " AND (expires_at IS NULL OR expires_at > ?)"
        params.append(as_of.isoformat())
    with _connect(db_path) as conn:
        rows = conn.execute(sql + " ORDER BY created_at, id", params).fetchall()
    return [_event_from_row(r) for r in rows]


# ---------- customer_notes ----------

def add_customer_note(db_path: Path, customer_id: str, note: str, as_of: date) -> CustomerNote:
    """Store a free-text note (max 500 chars). Raises ValueError if empty or too long."""
    if not note.strip():
        raise ValueError("note must not be empty")
    if len(note) > NOTE_MAX_CHARS:
        raise ValueError(f"note is {len(note)} chars; max is {NOTE_MAX_CHARS}")
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO customer_notes (customer_id, note, created_at) VALUES (?, ?, ?)",
            (customer_id, note, as_of.isoformat()),
        )
        new_id = cur.lastrowid
    return CustomerNote(id=new_id, customer_id=customer_id, note=note, created_at=as_of)


def list_customer_notes(db_path: Path, customer_id: str, as_of: date) -> list[CustomerNote]:
    """Notes for a customer created before or on as_of, oldest first."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM customer_notes WHERE customer_id = ? AND created_at <= ? ORDER BY created_at, id",
            (customer_id, as_of.isoformat()),
        ).fetchall()
    return [
        CustomerNote(id=r["id"], customer_id=r["customer_id"], note=r["note"], created_at=_to_date(r["created_at"]))
        for r in rows
    ]


# ---------- messages (short-term conversation memory) ----------

def add_message(
    db_path: Path, session_id: str, role: str, content: str, tool_calls_json: str | None, as_of: date
) -> MessageRecord:
    """Append one chat message to a session (role is validated before anything is written)."""
    record = MessageRecord(id=0, session_id=session_id, role=role, content=content,
                           tool_calls_json=tool_calls_json, created_at=as_of)
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, record.role, content, tool_calls_json, as_of.isoformat()),
        )
        record.id = cur.lastrowid
    return record


def list_messages(db_path: Path, session_id: str, limit: int = 10) -> list[MessageRecord]:
    """Last `limit` messages of a session, oldest first."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?", (session_id, limit)
        ).fetchall()
    return [
        MessageRecord(id=r["id"], session_id=r["session_id"], role=r["role"], content=r["content"],
                      tool_calls_json=r["tool_calls_json"], created_at=_to_date(r["created_at"]))
        for r in reversed(rows)
    ]


# ---------- deterministic policy (SPEC section 7) ----------

def compute_expires_at(outcome: str, reason: Reason | None, created_at: date) -> date | None:
    """Suppression end date for a declined event (30 or 90 days), else None."""
    if outcome != "declined" or reason not in SUPPRESSION_DAYS:
        return None
    return created_at + timedelta(days=SUPPRESSION_DAYS[reason])


def active_suppressions(db_path: Path, customer_id: str, as_of: date) -> dict[str, str]:
    """Map stock_code -> reason for products suppressed for this customer at as_of.

    If several active declines exist for one product, the one that expires last wins.
    """
    events = list_feedback_events(db_path, customer_id, as_of, include_expired=False)   # SQL drops expired ones
    latest: dict[str, FeedbackEventRecord] = {}
    for ev in events:
        if ev.expires_at is None or ev.stock_code is None:
            continue  # no expiry = not a suppressing decline (ordered, price); no stock_code = nothing to hide
        if ev.stock_code not in latest or ev.expires_at > latest[ev.stock_code].expires_at:
            latest[ev.stock_code] = ev
    return {code: f"declined: {ev.reason} (until {ev.expires_at.isoformat()})" for code, ev in latest.items()}


def is_price_sensitive(db_path: Path, customer_id: str, as_of: date) -> bool:
    """True if the customer ever declined for reason 'price' on or before as_of."""
    events = list_feedback_events(db_path, customer_id, as_of, include_expired=True)
    return any(ev.outcome == "declined" and ev.reason == "price" for ev in events)


def describe_effect(event: FeedbackEventRecord) -> str:
    """Plain-language policy effect of a stored event (shown to the rep and the agent)."""
    if event.expires_at is not None:
        if event.stock_code is None:
            return "recorded; no stock_code given, so nothing was suppressed"
        return f"{event.stock_code} suppressed until {event.expires_at.isoformat()}"
    if event.outcome == "declined" and event.reason == "price":
        return "recorded; customer marked price-sensitive (product not suppressed)"
    if event.outcome == "asked_alternative":
        return "recorded; saved as a customer note"
    return "recorded"


# ---------- high-level helpers (used by the API and the agent tools: one policy, one place) ----------

def record_feedback(db_path: Path, event: FeedbackEvent, as_of: date) -> WriteMemoryResult:
    """Store a feedback event, apply its side effects (asked_alternative -> note) and describe the effect."""
    before = list_feedback_events(db_path, event.customer_id, as_of, include_expired=True)
    record = add_feedback_event(db_path, event, as_of)
    is_new = all(ev.id != record.id for ev in before)
    if event.outcome == "asked_alternative" and is_new:
        product = event.stock_code or "an unspecified product"
        text = event.note or f"Asked for an alternative to {product}."
        add_customer_note(db_path, event.customer_id, text, as_of)
    effect = describe_effect(record)
    return WriteMemoryResult(event=record, effect=effect if is_new else f"duplicate ignored; {effect}")


def read_memory(db_path: Path, customer_id: str, as_of: date) -> MemoryResult:
    """Notes + non-expired feedback events + price-sensitive flag for one customer."""
    return MemoryResult(
        customer_id=customer_id,
        notes=list_customer_notes(db_path, customer_id, as_of),
        feedback_events=list_feedback_events(db_path, customer_id, as_of),
        price_sensitive=is_price_sensitive(db_path, customer_id, as_of),
    )


def apply_suppressions(prediction: PredictionResult, db_path: Path, as_of: date) -> PredictionResult:
    """Return a copy of `prediction` with suppressed items flagged (not removed) and price_sensitive set.

    Flagging instead of deleting keeps the decision visible: the agent can tell the rep *why* an item is missing.
    """
    suppressed = active_suppressions(db_path, prediction.customer_id, as_of)
    recs = [
        rec.model_copy(update={"suppressed": True, "suppressed_reason": suppressed[rec.stock_code]})
        if rec.stock_code in suppressed else rec
        for rec in prediction.recommendations
    ]
    return prediction.model_copy(update={
        "recommendations": recs,
        "price_sensitive": prediction.price_sensitive or is_price_sensitive(db_path, prediction.customer_id, as_of),
    })
