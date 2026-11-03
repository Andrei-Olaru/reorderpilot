"""Ranking and probability metrics: precision/recall/hit-rate@k, PR-AUC, ROC-AUC, Brier, abstention curve.

A "query" is one (cutoff, customer) pair: the same customer at two cutoffs counts twice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

QUERY = ["cutoff", "customer_id"]


def top_k(df: pd.DataFrame, score: pd.Series, k: int, eligible: pd.Series | None = None) -> pd.DataFrame:
    """Top-k rows per query by score (ties -> stock_code, deterministic). Ineligible rows are never picked."""
    d = df.assign(_score=score.to_numpy())
    if eligible is not None:
        d = d[eligible.to_numpy()]
    d = d.sort_values(QUERY + ["_score", "stock_code"], ascending=[True, True, False, True])
    return d.groupby(QUERY, sort=False).head(k)


def topk_metrics(df: pd.DataFrame, score: pd.Series, k: int = 5, eligible: pd.Series | None = None) -> dict:
    """precision@k (hits / k), recall@k, hit-rate@k, averaged over queries with >= 1 positive label."""
    n_pos = df.groupby(QUERY)["label"].sum()
    n_pos = n_pos[n_pos > 0]
    hits = top_k(df, score, k, eligible).groupby(QUERY)["label"].sum().reindex(n_pos.index, fill_value=0)
    return {
        f"precision@{k}": float((hits / k).mean()),
        f"recall@{k}": float((hits / n_pos).mean()),
        f"hit_rate@{k}": float((hits > 0).mean()),
        "n_queries": int(len(n_pos)),
    }


def pair_metrics(y: pd.Series, score: pd.Series, prob: bool = False) -> dict:
    """ROC-AUC and PR-AUC (average precision) on all pairs; Brier only when `score` is a probability."""
    out = {"roc_auc": float(roc_auc_score(y, score)), "pr_auc": float(average_precision_score(y, score)),
           "n_pairs": int(len(y)), "base_rate": float(np.mean(y))}
    if prob:
        out["brier"] = float(brier_score_loss(y, score))
    return out


def abstention_curve(df: pd.DataFrame, prob: pd.Series, taus: list[float], k: int = 5,
                     n_customers_total: int | None = None) -> pd.DataFrame:
    """For each tau: recommend top-k items with p >= tau; a query with none is an abstention.

    coverage = covered queries / all queries (pass n_customers_total to also count cold-start customers),
    precision = positives among recommended items, recall = recommended positives / all positives.
    """
    total = n_customers_total or df.groupby(QUERY).ngroups
    ranked = top_k(df, prob, k)
    n_pos_all = int(df["label"].sum())
    rows = []
    for tau in taus:
        rec = ranked[ranked["_score"] >= tau]
        n_rec = len(rec)
        tp = int(rec["label"].sum())
        precision = tp / n_rec if n_rec else float("nan")
        recall = tp / n_pos_all if n_pos_all else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if n_rec and tp else 0.0
        rows.append({"tau": tau, "coverage": rec.groupby(QUERY).ngroups / total, "precision": precision,
                     "recall": recall, "f1": f1, "n_recommended": n_rec})
    return pd.DataFrame(rows)


def calibration_table(y: pd.Series, prob: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed rate in quantile bins of the predicted probability."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(prob)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    return d.groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()
