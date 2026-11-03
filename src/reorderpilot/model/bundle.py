"""ModelBundle (model + calibrator + tau) and calibrators. Own module so pickles don't point at __main__."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Platt scaling: logistic regression on the logit of the raw score. Same interface as IsotonicRegression."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        self.lr = LogisticRegression().fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(_logit(np.asarray(p)).reshape(-1, 1))[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def make_calibrator(kind: str):
    """'isotonic' (monotone step function) or 'platt' (sigmoid)."""
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0) if kind == "isotonic" else PlattCalibrator()


@dataclass
class ModelBundle:
    """What predict.py needs: the raw model, the calibrator, the abstention threshold, and the feature order."""

    model: HistGradientBoostingClassifier
    calibrator: object
    tau: float
    features: list[str]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(df[self.features].astype(float))[:, 1]
        return np.asarray(self.calibrator.predict(raw), dtype=float)
