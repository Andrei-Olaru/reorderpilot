"""SQLite memory store: feedback events, customer notes, chat messages, and the deterministic policy.

Every function takes `db_path` explicitly so tests can use a temporary file (never the real memory.db).
Dates are simulated (`as_of`), never wall-clock. Feedback events are append-only (audit trail).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from reorderpilot.api.schemas import CustomerNote, FeedbackEvent, FeedbackEventRecord, MessageRecord, Reason


def init_db(db_path: Path) -> None:
    """Create the three tables if they do not exist."""
    raise NotImplementedError("M2 task B1")


# ---------- feedback_events ----------

def add_feedback_event(db_path: Path, event: FeedbackEvent, as_of: date) -> FeedbackEventRecord:
    """Store a feedback event with created_at=as_of and expires_at from the policy."""
    raise NotImplementedError("M2 task B1")


def list_feedback_events(
    db_path: Path, customer_id: str, as_of: date, include_expired: bool = False
) -> list[FeedbackEventRecord]:
    """Events for a customer created before as_of, optionally including expired ones."""
    raise NotImplementedError("M2 task B1")


# ---------- customer_notes ----------

def add_customer_note(db_path: Path, customer_id: str, note: str, as_of: date) -> CustomerNote:
    """Store a free-text note (max 500 chars)."""
    raise NotImplementedError("M2 task B1")


def list_customer_notes(db_path: Path, customer_id: str, as_of: date) -> list[CustomerNote]:
    """Notes for a customer created before or on as_of, oldest first."""
    raise NotImplementedError("M2 task B1")


# ---------- messages (short-term conversation memory) ----------

def add_message(
    db_path: Path, session_id: str, role: str, content: str, tool_calls_json: str | None, as_of: date
) -> MessageRecord:
    """Append one chat message to a session."""
    raise NotImplementedError("M2 task B1")


def list_messages(db_path: Path, session_id: str, limit: int = 10) -> list[MessageRecord]:
    """Last `limit` messages of a session, oldest first."""
    raise NotImplementedError("M2 task B1")


# ---------- deterministic policy (SPEC section 7) ----------

def compute_expires_at(outcome: str, reason: Reason | None, created_at: date) -> date | None:
    """Suppression end date for a declined event (30 or 90 days), else None."""
    raise NotImplementedError("M2 task B2")


def active_suppressions(db_path: Path, customer_id: str, as_of: date) -> dict[str, str]:
    """Map stock_code -> reason for products suppressed for this customer at as_of."""
    raise NotImplementedError("M2 task B2")


def is_price_sensitive(db_path: Path, customer_id: str, as_of: date) -> bool:
    """True if the customer ever declined for reason 'price' before as_of."""
    raise NotImplementedError("M2 task B2")
