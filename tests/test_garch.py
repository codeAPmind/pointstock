"""GARCH on simulated GJR-GARCH-t data recovers the true parameters."""
import math

import numpy as np
import pandas as pd
import pytest

from premarket import garch as G

TRUE = G.GarchParams(mu=0.05, omega=0.05, alpha=0.05, gamma=0.10, beta=0.85, nu=6.0)
TOL = {"mu": 0.04, "omega": 0.04, "alpha": 0.035, "gamma": 0.06, "beta": 0.05, "nu": 2.5}


@pytest.fixture(scope="module")
def sim():
    return G.simulate(6000, TRUE, seed=42)


def _check(p: G.GarchParams):
    for name, tol in TOL.items():
        est, true = getattr(p, name), getattr(TRUE, name)
        assert abs(est - true) <= tol, f"{name}: estimated {est:.4f}, true {true:.4f}, tol {tol}"


def test_arch_recovers_parameters(sim):
    _check(G.fit_arch(sim))


def test_scipy_fallback_recovers_parameters(sim):
    _check(G.fit_scipy(sim))


def test_arch_and_scipy_agree(sim):
    a, s = G.fit_arch(sim), G.fit_scipy(sim)
    assert abs(a.persistence - s.persistence) < 0.01
    assert abs(G.neg_loglik(np.array([*vars(a).values()]), sim)
               - G.neg_loglik(np.array([*vars(s).values()]), sim)) < 2.0


def test_simulated_data_has_target_unconditional_vol(sim):
    assert abs(np.std(sim) - TRUE.long_run_sigma) / TRUE.long_run_sigma < 0.1


def test_forecast_bands_and_coverage_on_simulated_prices(sim):
    close = pd.Series(100 * np.exp(np.cumsum(sim) / 100),
                      index=pd.bdate_range("2003-01-01", periods=len(sim)))
    fc = G.forecast(close, window=1000)
    assert fc.band95[0] < fc.band68[0] < fc.last_close < fc.band68[1] < fc.band95[1]
    assert fc.regime in {"HIGH", "NORMAL", "LOW"}
    cov = G.band_coverage(close, window=1000, test_days=1000)
    assert abs(cov.inside68 - 0.68) < 0.05
    assert abs(cov.inside95 - 0.95) < 0.03


def test_regime_label():
    assert G.regime_label(2.0, 1.0) == "HIGH"
    assert G.regime_label(0.5, 1.0) == "LOW"
    assert G.regime_label(1.0, 1.0) == "NORMAL"
    assert G.regime_label(1.0, math.inf) == "HIGH"
