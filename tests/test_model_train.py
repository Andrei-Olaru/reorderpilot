"""Training pieces on synthetic data: model fits, calibration choice runs, bundle outputs valid probabilities."""

import numpy as np
import pandas as pd

from reorderpilot.features import FEATURE_COLUMNS
from reorderpilot.model.bundle import ModelBundle, make_calibrator
from reorderpilot.model.train import choose_calibration, fit_hgb


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df["label"] = (df["overdue_ratio"] + rng.normal(scale=0.5, size=n) > 1.0).astype(int)
    df["cutoff"] = np.where(np.arange(n) < n // 2, "2011-07-01", "2011-08-01")
    return df


def test_fit_calibrate_and_predict_probabilities():
    df = _synthetic()
    model = fit_hgb(df, {"learning_rate": 0.1, "max_leaf_nodes": 7, "max_iter": 20})
    raw = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    kind, scores = choose_calibration(df, raw)
    assert kind in {"isotonic", "platt"} and set(scores) == {"isotonic", "platt", "uncalibrated"}

    bundle = ModelBundle(model, make_calibrator(kind).fit(raw, df["label"].to_numpy()), 0.1, list(FEATURE_COLUMNS))
    p = bundle.predict_proba(df)
    assert p.shape == (len(df),) and np.all((p >= 0) & (p <= 1))
    # The signal is real, so higher overdue_ratio must get higher average probability.
    assert p[df["overdue_ratio"] > 1.5].mean() > p[df["overdue_ratio"] < 0].mean()
