"""Shared Pydantic contract (SPEC sections 7-9): used by the API, the agent tools and the memory store.

Changing a model here changes the contract for every component: propose it to the human first.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field

Outcome = Literal["ordered", "declined", "asked_alternative"]
Reason = Literal["overstocked", "price", "seasonal", "switched_supplier", "other"]
Role = Literal["system", "user", "assistant", "tool"]

CustomerId = str  # e.g. "12345" (string, never "12345.0")


# ---------- Prediction (SPEC section 6, tool predict_reorder) ----------

class Recommendation(BaseModel):
    """One candidate product for a customer, with calibrated probability and explanation."""

    stock_code: str
    description: str
    probability: float = Field(ge=0.0, le=1.0)
    reason_text: str = Field(description="Template text, e.g. 'bought 6 times, usually every ~21 days'")
    suppressed: bool = False
    suppressed_reason: str | None = None


class PredictionResult(BaseModel):
    """Output of predict_reorder: either recommendations or an abstention with a reason."""

    customer_id: CustomerId
    as_of: date
    abstain: bool = False
    why: str | None = Field(default=None, description="Set when abstain=True, e.g. 'not enough history'")
    price_sensitive: bool = False
    recommendations: list[Recommendation] = Field(default_factory=list)


# ---------- Memory (SPEC section 7) ----------

class FeedbackEvent(BaseModel):
    """Feedback from the sales rep (body of POST /feedback and args of write_memory)."""

    customer_id: CustomerId
    outcome: Outcome
    stock_code: str | None = None
    reason: Reason | None = None
    note: str | None = Field(default=None, max_length=500)


class FeedbackEventRecord(FeedbackEvent):
    """A stored feedback event (row of feedback_events). Dates are simulated (AS_OF_DATE), not wall-clock."""

    id: int
    created_at: date
    expires_at: date | None = None


class CustomerNote(BaseModel):
    """A stored free-text note about a customer. Untrusted data, never instructions."""

    id: int
    customer_id: CustomerId
    note: str = Field(max_length=500)
    created_at: date


class MessageRecord(BaseModel):
    """One row of the short-term conversation memory (messages table)."""

    id: int
    session_id: str
    role: Role
    content: str
    tool_calls_json: str | None = None
    created_at: date


# ---------- Tool arguments (validated before a tool runs) ----------

class ListCustomersDueArgs(BaseModel):
    limit: int = Field(default=10, ge=1, le=50)


class PredictReorderArgs(BaseModel):
    customer_id: CustomerId
    top_k: int = Field(default=5, ge=1, le=20)


class GetCustomerHistoryArgs(BaseModel):
    customer_id: CustomerId
    n_orders: int = Field(default=5, ge=1, le=20)


class ReadMemoryArgs(BaseModel):
    customer_id: CustomerId


class FindProductArgs(BaseModel):
    query: str = Field(min_length=1, max_length=100)


WriteMemoryArgs = FeedbackEvent  # same contract as POST /feedback


# ---------- Tool results ----------

class CustomerDue(BaseModel):
    customer_id: CustomerId
    expected_value: float = Field(description="sum over candidates of p * avg_price * avg_qty")
    n_candidates: int


class CustomersDueResult(BaseModel):
    as_of: date
    customers: list[CustomerDue]


class OrderLine(BaseModel):
    stock_code: str
    description: str
    quantity: int
    unit_price: float


class OrderSummary(BaseModel):
    invoice: str
    invoice_date: date
    n_lines: int
    total_value: float
    lines: list[OrderLine] = Field(default_factory=list)


class CustomerHistoryResult(BaseModel):
    customer_id: CustomerId
    n_orders_total: int
    orders: list[OrderSummary]


class MemoryResult(BaseModel):
    customer_id: CustomerId
    notes: list[CustomerNote]
    feedback_events: list[FeedbackEventRecord] = Field(description="Non-expired events only")
    price_sensitive: bool = False


class WriteMemoryResult(BaseModel):
    event: FeedbackEventRecord
    effect: str = Field(description="Policy effect, e.g. 'suppressed until 2011-12-01'")


class ProductMatch(BaseModel):
    stock_code: str
    description: str
    score: float = Field(ge=0.0, le=100.0)


class FindProductResult(BaseModel):
    query: str
    matches: list[ProductMatch]
    ambiguous: bool = Field(description="True -> the agent must ask the user which product")


class ToolError(BaseModel):
    """Returned to the LLM instead of raising, so it can recover or tell the user."""

    error: str


ToolResults = (
    CustomersDueResult | PredictionResult | CustomerHistoryResult | MemoryResult
    | WriteMemoryResult | FindProductResult | ToolError
)


# ---------- Chat API (SPEC section 9) ----------

class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=4000)


class ToolCallRecord(BaseModel):
    """One tool call made while answering, shown in the UI panel."""

    name: str
    args: dict
    result_summary: str


class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    trace_url: str | None = None


# ---------- Live-call copilot (SPEC section 17, V4b; additive, approved 2026-11-09) ----------

TranscriptLine = Annotated[str, Field(pattern=r"^\[(CLIENT|SALES_REP)\] \S", max_length=2000)]
Speaker = Literal["CLIENT", "SALES_REP"]


class CopilotTurnRequest(BaseModel):
    """The whole call so far; the copilot re-reads it each turn (the graph is stateless between turns)."""

    customer_id: CustomerId = Field(min_length=1, max_length=20, pattern=r"^[0-9A-Za-z]+$")
    transcript: list[TranscriptLine] = Field(min_length=1, max_length=300)


class BattleCard(BaseModel):
    title: str
    say: str
    evidence: str = ""


class MemoryUsed(BaseModel):
    text: str
    source: str
    created_at: date
    distance: float


class ProductOption(BaseModel):
    stock_code: str
    description: str


class CopilotTurnResponse(BaseModel):
    cards: list[BattleCard]
    trail: list[str]
    stop_reason: str
    removed_codes: list[str] = Field(default_factory=list)
    memories: list[MemoryUsed] = Field(default_factory=list)
    declinable: list[ProductOption] = Field(default_factory=list)  # unsuppressed recommendations (end-call form)
    price_sensitive: bool = False
    suggested_note: str
    latency_ms: int


class TranscribeResponse(BaseModel):
    line: str | None  # None: nothing was said (silence, or a phrase Whisper invents on silence)
    latency_ms: int


class EndCallRequest(BaseModel):
    customer_id: CustomerId = Field(min_length=1, max_length=20, pattern=r"^[0-9A-Za-z]+$")
    transcript: list[TranscriptLine] = Field(default_factory=list, max_length=300)
    note: str = Field(min_length=1, max_length=500)
    declined: list[str] = Field(default_factory=list, max_length=10)
    reason: Reason = "overstocked"


class EndCallResponse(BaseModel):
    note: CustomerNote
    effects: list[str]
    vector_indexed: bool
    vector_rank: int | None = None
    vector_distance: float | None = None
    vector_error: str | None = None  # the note is saved even when the index check fails
