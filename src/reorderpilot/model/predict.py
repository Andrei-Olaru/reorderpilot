"""Load the trained model and score one customer's candidate products at the simulated date.

Returns raw model output only. Memory suppression is applied later, in agent/tools.py (SPEC section 7).
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from reorderpilot.api.schemas import PredictionResult, Recommendation
from reorderpilot.config import ARTIFACTS_DIR, MIN_PRIOR_ORDERS, get_settings
from reorderpilot.data.loader import load_transactions
from reorderpilot.features import build_features
from reorderpilot.model.bundle import ModelBundle


@lru_cache(maxsize=1)
def load_bundle(path: Path = ARTIFACTS_DIR / "model.joblib") -> ModelBundle:
    """Load model + calibrator + tau saved by `python -m reorderpilot.model.train`."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m reorderpilot.model.train")
    return joblib.load(path)


def explain(row: pd.Series) -> str:
    """Template explanation from feature values (no SHAP), e.g. 'bought 6 times, usually every ~21 days, ...'."""
    n, days = int(row["n_orders_p"]), int(row["days_since_last_p"])
    ago = f"{days} day{'' if days == 1 else 's'} ago"
    if n == 1 or pd.isna(row["mean_interval_p"]):
        return f"bought once, {ago}"
    text = f"bought {n} times, usually every ~{round(row['mean_interval_p'])} days, last bought {ago}"
    if row["overdue_ratio"] >= 1.0:
        text += " (due now)"
    return text


def score_candidates(tx: pd.DataFrame, bundle: ModelBundle, as_of: date,
                     customer_ids: list[str] | None = None) -> pd.DataFrame:
    """Features + calibrated `probability` for every candidate pair (all customers, or only customer_ids)."""
    feats = build_features(tx, as_of, customer_ids)
    if feats.empty:
        return feats.assign(probability=pd.Series(dtype=float))
    return feats.assign(probability=bundle.predict_proba(feats))


def predict_for(tx: pd.DataFrame, bundle: ModelBundle, customer_id: str, top_k: int, as_of: date) -> PredictionResult:
    """Core logic of predict_reorder with data and model passed in (easy to test)."""
    def abstain(why: str) -> PredictionResult:
        return PredictionResult(customer_id=customer_id, as_of=as_of, abstain=True, why=why)

    feats = build_features(tx, as_of, [customer_id])
    if feats.empty:
        return abstain(f"no purchases before {as_of}")
    n_orders = int(feats["cust_n_orders"].iloc[0])
    if n_orders < MIN_PRIOR_ORDERS:
        return abstain(f"not enough history: {n_orders} prior order(s), need at least {MIN_PRIOR_ORDERS}")

    feats = feats.assign(probability=bundle.predict_proba(feats))
    feats = feats.sort_values(["probability", "stock_code"], ascending=[False, True])
    best = float(feats["probability"].iloc[0])
    if best < bundle.tau:
        return abstain(f"low confidence: best product is {best:.0%}, below the {bundle.tau:.0%} threshold")

    descriptions = tx.drop_duplicates("stock_code").set_index("stock_code")["description"]
    recs = [
        Recommendation(stock_code=r["stock_code"], description=str(descriptions.get(r["stock_code"], "")),
                       probability=round(float(r["probability"]), 4), reason_text=explain(r))
        for _, r in feats[feats["probability"] >= bundle.tau].head(top_k).iterrows()
    ]
    return PredictionResult(customer_id=customer_id, as_of=as_of, recommendations=recs)


def predict_reorder(customer_id: str, top_k: int = 5, as_of: date | None = None) -> PredictionResult:
    """Return the top_k re-order candidates for a customer, or an abstention (cold start / below tau).

    `as_of` defaults to Settings.as_of_date; features use only data strictly before it.
    """
    as_of = as_of or get_settings().as_of_date
    return predict_for(load_transactions(), load_bundle(), str(customer_id), top_k, as_of)
