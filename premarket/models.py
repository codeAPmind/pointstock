"""Direction models with walk-forward validation (DESIGN.md, "Direction models").

Four shallow classifiers predict P(next close > today's close). Each one is
scored only on days it never saw: train on every day before a 21-day test
window, predict the window, slide forward, retrain. Skill is the Brier skill
score against a no-skill baseline that always predicts the up-rate of the
training data available at that time:

    BSS = 1 - Brier(model) / Brier(baseline)        BSS <= 0 means no edge

Ensemble weights are proportional to max(0, BSS). If no model has BSS > 0
the ensemble returns None and the report says "no direction edge".

Never random K-fold: Platt scaling for the SVM is calibrated with a
TimeSeriesSplit inside the training window only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import feature_columns

log = logging.getLogger(__name__)

STEP = 21            # trading days per walk-forward window
TEST_DAYS = 750      # about 3 years scored out-of-sample
MIN_TRAIN = 250      # never fit on less than a year
MODEL_VERSION = 1    # bump when model settings or features change


def make_models(seed: int = 0) -> dict[str, Callable[[], BaseEstimator]]:
    """Factories so every fold starts from a fresh, unfitted model."""
    return {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=40, n_jobs=-1, random_state=seed),
        "svm_rbf": lambda: make_pipeline(StandardScaler(), CalibratedClassifierCV(
            SVC(C=0.5, kernel="rbf", gamma="scale"), method="sigmoid", cv=TimeSeriesSplit(3))),
        "naive_bayes": lambda: make_pipeline(StandardScaler(), GaussianNB()),
    }


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def bss(p: np.ndarray, p_base: np.ndarray, y: np.ndarray) -> float:
    b = brier(p_base, y)
    return 1 - brier(p, y) / b if b > 0 else 0.0


def training_frame(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Rows with every feature and a known target. Columns that are NaN for most
    of history (e.g. a short-lived market series) are dropped first."""
    cols = feature_columns(table)
    cols = [c for c in cols if table[c].notna().mean() > 0.9]
    df = table[cols + ["target_up"]].replace([np.inf, -np.inf], np.nan).dropna()
    return df[cols], df["target_up"].astype(int), cols


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #

@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp     # last training date
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def folds(index: pd.DatetimeIndex, test_days: int = TEST_DAYS, step: int = STEP,
          min_train: int = MIN_TRAIN) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train positions, test positions) pairs; training is always strictly earlier."""
    n = len(index)
    start = max(min_train, n - test_days)
    out = []
    for s in range(start, n, step):
        out.append((np.arange(0, s), np.arange(s, min(s + step, n))))
    return out


@dataclass
class ModelScore:
    model: str
    n_test: int
    accuracy: float
    brier: float
    brier_base: float
    bss: float


@dataclass
class WalkForwardResult:
    ticker: str
    scores: dict[str, ModelScore]
    predictions: pd.DataFrame          # per test day: y, base, one column per model
    folds: list[Fold]
    test_start: str
    test_end: str

    @property
    def best(self) -> ModelScore:
        return max(self.scores.values(), key=lambda s: s.bss)

    def weights(self) -> dict[str, float]:
        return ensemble_weights({k: s.bss for k, s in self.scores.items()})


def walk_forward(X: pd.DataFrame, y: pd.Series, ticker: str = "",
                 factories: dict[str, Callable[[], BaseEstimator]] | None = None,
                 test_days: int = TEST_DAYS, step: int = STEP, min_train: int = MIN_TRAIN,
                 ) -> WalkForwardResult:
    factories = factories or make_models()
    preds = {k: [] for k in factories}
    base, idx, fold_meta = [], [], []
    for tr, te in folds(X.index, test_days, step, min_train):
        Xtr, ytr, Xte = X.iloc[tr], y.iloc[tr], X.iloc[te]
        fold_meta.append(Fold(X.index[tr[0]], X.index[tr[-1]], X.index[te[0]], X.index[te[-1]]))
        base.extend([ytr.mean()] * len(te))        # no-skill: training up-rate
        idx.extend(X.index[te])
        for name, make in factories.items():
            m = make().fit(Xtr, ytr)
            preds[name].extend(m.predict_proba(Xte)[:, 1])
    pred = pd.DataFrame({"y": y.loc[idx].to_numpy(), "base": base, **preds},
                        index=pd.DatetimeIndex(idx, name="date"))
    scores = {}
    for name in factories:
        p = pred[name].to_numpy()
        scores[name] = ModelScore(
            model=name, n_test=len(pred), accuracy=float(np.mean((p > 0.5) == pred["y"])),
            brier=brier(p, pred["y"]), brier_base=brier(pred["base"], pred["y"]),
            bss=bss(p, pred["base"], pred["y"]))
    return WalkForwardResult(ticker, scores, pred, fold_meta,
                             str(pred.index[0].date()), str(pred.index[-1].date()))


def ensemble_weights(bss_by_model: dict[str, float]) -> dict[str, float]:
    """Weights proportional to max(0, BSS); all zero when no model has skill."""
    pos = {k: max(0.0, v) for k, v in bss_by_model.items()}
    tot = sum(pos.values())
    return {k: (v / tot if tot > 0 else 0.0) for k, v in pos.items()}


def ensemble_prob(probs: dict[str, float], weights: dict[str, float]) -> float | None:
    if sum(weights.values()) <= 0:
        return None
    return float(sum(weights[k] * probs[k] for k in probs))


# --------------------------------------------------------------------------- #
# Saved bundle (weekly refit) and morning prediction
# --------------------------------------------------------------------------- #

@dataclass
class ModelBundle:
    ticker: str
    fitted_on: str               # ISO date of the refit
    last_train_date: str
    version: int
    feature_cols: list[str]
    models: dict                 # name -> fitted estimator (trained on all labelled rows)
    scores: dict[str, ModelScore]
    weights: dict[str, float]
    up_rate: float               # training up-rate (no-skill baseline)
    test_start: str
    test_end: str
    n_folds: int


def fit_bundle(table: pd.DataFrame, ticker: str, **wf_kwargs) -> tuple[ModelBundle, WalkForwardResult]:
    X, y, cols = training_frame(table)
    wf = walk_forward(X, y, ticker, **wf_kwargs)
    factories = wf_kwargs.get("factories") or make_models()
    final = {k: f().fit(X, y) for k, f in factories.items()}
    bundle = ModelBundle(
        ticker=ticker, fitted_on=date.today().isoformat(), last_train_date=str(X.index[-1].date()),
        version=MODEL_VERSION, feature_cols=cols, models=final, scores=wf.scores,
        weights=wf.weights(), up_rate=float(y.mean()), test_start=wf.test_start,
        test_end=wf.test_end, n_folds=len(wf.folds))
    return bundle, wf


def bundle_path(data_dir: str | Path, ticker: str) -> Path:
    return Path(data_dir) / "models" / f"{ticker}.joblib"


def save_bundle(bundle: ModelBundle, data_dir: str | Path) -> Path:
    import joblib

    p = bundle_path(data_dir, bundle.ticker)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, p)
    return p


def load_bundle(data_dir: str | Path, ticker: str) -> ModelBundle | None:
    import joblib

    p = bundle_path(data_dir, ticker)
    if not p.exists():
        return None
    try:
        b = joblib.load(p)
    except Exception as exc:
        log.warning("could not load %s: %s", p, exc)
        return None
    return b if getattr(b, "version", None) == MODEL_VERSION else None


@dataclass
class DirectionForecast:
    probs: dict[str, float]          # today's P(up) per model
    weights: dict[str, float]
    p_ml: float | None               # ensemble, None = no direction edge
    up_rate: float
    scores: dict[str, ModelScore]
    fitted_on: str
    test_start: str
    test_end: str
    age_days: int
    warnings: list[str] = field(default_factory=list)

    @property
    def best(self) -> ModelScore:
        return max(self.scores.values(), key=lambda s: s.bss)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["best"] = asdict(self.best)
        return d


def predict_today(bundle: ModelBundle, table: pd.DataFrame, today: date | None = None) -> DirectionForecast:
    """Score the last row (features at the last close) with the saved models."""
    warnings = []
    missing = [c for c in bundle.feature_cols if c not in table.columns]
    if missing:
        raise ValueError(f"feature columns missing for {bundle.ticker}: {missing}")
    row = table[bundle.feature_cols].iloc[[-1]].replace([np.inf, -np.inf], np.nan)
    if row.isna().any(axis=None):
        bad = list(row.columns[row.isna().iloc[0]])
        warnings.append(f"NaN features on the last row, filled from the previous day: {bad}")
        row = table[bundle.feature_cols].ffill().iloc[[-1]]
    probs = {k: float(m.predict_proba(row)[0, 1]) for k, m in bundle.models.items()}
    age = ((today or date.today()) - date.fromisoformat(bundle.fitted_on)).days
    if age > 8:
        warnings.append(f"models are {age} days old (weekly refit missed)")
    return DirectionForecast(probs=probs, weights=bundle.weights,
                             p_ml=ensemble_prob(probs, bundle.weights), up_rate=bundle.up_rate,
                             scores=bundle.scores, fitted_on=bundle.fitted_on,
                             test_start=bundle.test_start, test_end=bundle.test_end,
                             age_days=age, warnings=warnings)


def format_table(wf_or_bundle) -> str:
    """Plain-text walk-forward table (accuracy, Brier, BSS per model)."""
    s = wf_or_bundle.scores
    lines = [f"{'model':<14}{'n':>6}{'accuracy':>10}{'brier':>9}{'base':>9}{'BSS':>9}{'weight':>8}"]
    w = ensemble_weights({k: v.bss for k, v in s.items()})
    for k, v in s.items():
        lines.append(f"{k:<14}{v.n_test:>6}{v.accuracy:>10.1%}{v.brier:>9.4f}{v.brier_base:>9.4f}"
                     f"{v.bss:>+9.4f}{w[k]:>8.2f}")
    return "\n".join(lines)
