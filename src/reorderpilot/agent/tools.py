"""Agent tools (SPEC section 8): thin wrappers over model, history and memory, with Pydantic-validated args.

This is where memory suppression is applied to predict_reorder output, so it is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from reorderpilot.api.schemas import (
    CustomerHistoryResult,
    CustomersDueResult,
    FindProductArgs,
    FindProductResult,
    GetCustomerHistoryArgs,
    ListCustomersDueArgs,
    MemoryResult,
    PredictionResult,
    PredictReorderArgs,
    ReadMemoryArgs,
    ToolResults,
    WriteMemoryArgs,
    WriteMemoryResult,
)


@dataclass(frozen=True)
class ToolContext:
    """What every tool needs besides its args. memory_enabled=False is used by the memory ablation (eval B)."""

    as_of: date
    db_path: Path
    memory_enabled: bool = True


def list_customers_due(args: ListCustomersDueArgs, ctx: ToolContext) -> CustomersDueResult:
    """Customers ranked by expected re-order value, excluding abstained ones."""
    raise NotImplementedError("M3 task D1")


def predict_reorder(args: PredictReorderArgs, ctx: ToolContext) -> PredictionResult:
    """Model predictions with memory suppression and price-sensitivity applied."""
    raise NotImplementedError("M3 task D1")


def get_customer_history(args: GetCustomerHistoryArgs, ctx: ToolContext) -> CustomerHistoryResult:
    """Summary of the customer's last n orders before as_of."""
    raise NotImplementedError("M3 task D1")


def read_memory(args: ReadMemoryArgs, ctx: ToolContext) -> MemoryResult:
    """Notes + non-expired feedback events (empty when memory is disabled)."""
    raise NotImplementedError("M3 task D1")


def write_memory(args: WriteMemoryArgs, ctx: ToolContext) -> WriteMemoryResult:
    """Store a feedback event and report its policy effect."""
    raise NotImplementedError("M3 task D1")


def find_product(args: FindProductArgs, ctx: ToolContext) -> FindProductResult:
    """Fuzzy-match a product name to stock codes; flags ambiguous matches."""
    raise NotImplementedError("M3 task D1")


def get_tool_specs() -> list[dict]:
    """OpenAI-format tool definitions (JSON schema built from the Args models)."""
    raise NotImplementedError("M3 task D1")


def execute_tool(name: str, raw_args: str, ctx: ToolContext) -> ToolResults:
    """Parse + validate JSON args, run the named tool, return a result or ToolError (never raises)."""
    raise NotImplementedError("M3 task D1")
