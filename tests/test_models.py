"""Walk-forward honesty, Brier skill, ensemble weights, saved bundles."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from premarket import models as M
from premarket.features import TARGET_COLS, build_features
from tests.conftest import synthetic_ohlcv


class Recorder(BaseEstimator, ClassifierMixin):
    """Predicts the training up-rate and records which dates it was trained on."""
    log: list = []

    def fit(self, X, y):
        Recorder.log.append(("fit", X.index.min(), X.index.max()))
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        Recorder.log.append(("predict", X.index.min(), X.index.max()))
        return np.column_stack([np.full(len(X), 1 - self.p_), np.full(len(X), self.p_)])


@pytest.fixture(scope="module")
def table():
    return build_features(synthetic_ohlcv(n=1300, seed=11))


def test_walk_forward_never_trains_on_or_after_test_date(table):
    X, y, _ = M.training_frame(table)
    Recorder.log = []
    wf = M.walk_forward(X, y, factories={"rec": Recorder}, test_days=400, step=21)
    events = Recorder.log
    assert len(events) == 2 * len(wf.folds)
    for (kind_f, _, train_max), (kind_p, test_min, test_max) in zip(events[::2], events[1::2]):
        assert kind_f == "fit" and kind_p == "predict"
        assert train_max < test_min, f"trained through {train_max} but tested from {test_min}"
    for f in wf.folds:
        assert f.train_end < f.test_start
    # Test windows tile the last 400 days with no gaps or overlaps.
    assert len(wf.predictions) == 400
    assert wf.predictions.index.is_unique and wf.predictions.index.is_monotonic_increasing


def test_baseline_uses_only_training_up_rate(table):
    """A model that predicts the training up-rate has exactly zero skill."""
    X, y, _ = M.training_frame(table)
    wf = M.walk_forward(X, y, factories={"rec": Recorder}, test_days=300)
    assert wf.scores["rec"].bss == pytest.approx(0.0, abs=1e-12)
    for f in wf.folds:
        rows = wf.predictions.loc[f.test_start:f.test_end, "base"]
        assert np.allclose(rows, y.loc[:f.train_end].mean())


def test_targets_never_used_as_features(table):
    _, _, cols = M.training_frame(table)
    assert not set(cols) & set(TARGET_COLS)


def test_brier_and_bss():
    y = np.array([1, 0, 1, 1])
    assert M.brier(np.array([1, 0, 1, 1]), y) == 0
    assert M.brier(np.full(4, 0.5), y) == 0.25
    assert M.bss(y.astype(float), np.full(4, 0.5), y) == 1.0
    assert M.bss(np.full(4, 0.5), np.full(4, 0.5), y) == 0.0


def test_ensemble_weights_and_no_edge():
    w = M.ensemble_weights({"a": 0.02, "b": 0.01, "c": -0.05})
    assert w == pytest.approx({"a": 2 / 3, "b": 1 / 3, "c": 0.0})
    assert M.ensemble_prob({"a": 0.6, "b": 0.3, "c": 0.9}, w) == pytest.approx(0.5)
    none = M.ensemble_weights({"a": -0.01, "b": 0.0})
    assert sum(none.values()) == 0
    assert M.ensemble_prob({"a": 0.7, "b": 0.6}, none) is None


def test_no_skill_on_pure_noise(table):
    """On a random walk the real models must not show a meaningful edge."""
    X, y, _ = M.training_frame(table)
    wf = M.walk_forward(X, y, test_days=420, step=42)
    for s in wf.scores.values():
        assert s.bss < 0.03, f"{s.model} shows BSS {s.bss:.3f} on noise — leakage?"


def test_bundle_roundtrip_and_predict(table, tmp_path):
    bundle, _ = M.fit_bundle(table, "TEST", test_days=200, step=50)
    M.save_bundle(bundle, tmp_path)
    loaded = M.load_bundle(tmp_path, "TEST")
    assert loaded is not None and loaded.feature_cols == bundle.feature_cols
    fc = M.predict_today(loaded, table)
    assert set(fc.probs) == set(M.make_models())
    assert all(0 < p < 1 for p in fc.probs.values())
    assert (fc.p_ml is None) == (sum(fc.weights.values()) == 0)


def test_stale_bundle_version_is_ignored(table, tmp_path, monkeypatch):
    bundle, _ = M.fit_bundle(table, "TEST", factories={"rec": Recorder}, test_days=100, step=50)
    M.save_bundle(bundle, tmp_path)
    monkeypatch.setattr(M, "MODEL_VERSION", M.MODEL_VERSION + 1)
    assert M.load_bundle(tmp_path, "TEST") is None
