"""Load the trained model and score one customer's candidate products at the simulated date.

Returns raw model output only. Memory suppression is applied later, in agent/tools.py (SPEC section 7).
"""

from __future__ import annotations

from datetime import date

from reorderpilot.api.schemas import PredictionResult


def predict_reorder(customer_id: str, top_k: int = 5, as_of: date | None = None) -> PredictionResult:
    """Return the top_k re-order candidates for a customer, or an abstention (cold start / below tau).

    `as_of` defaults to Settings.as_of_date; features use only data strictly before it.
    """
    raise NotImplementedError("M1 task A5")
