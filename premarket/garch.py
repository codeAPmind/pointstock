"""GJR-GARCH(1,1) with Student-t errors: the size of tomorrow's move.

    sigma_t^2 = omega + (alpha + gamma * I[eps_{t-1} < 0]) * eps_{t-1}^2 + beta * sigma_{t-1}^2

Returns are modelled in percent (r * 100) for numerical stability. The `arch`
package does the fit; a small scipy implementation of the same likelihood is
kept as a fallback (and lets the tests cross-check the two).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

log = logging.getLogger(__name__)

PARAM_NAMES = ["mu", "omega", "alpha", "gamma", "beta", "nu"]


@dataclass
class GarchParams:
    mu: float      # mean daily return, percent
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float      # Student-t degrees of freedom

    @property
    def persistence(self) -> float:
        return self.alpha + self.gamma / 2 + self.beta

    @property
    def long_run_sigma(self) -> float:
        """Unconditional daily sigma in percent (inf if non-stationary)."""
        p = self.persistence
        return math.sqrt(self.omega / (1 - p)) if p < 1 else float("inf")


@dataclass
class GarchForecast:
    params: GarchParams
    engine: str              # "arch" or "scipy"
    n_obs: int
    sigma_next: float        # next-day sigma, percent
    long_run_sigma: float    # percent
    regime: str              # HIGH / NORMAL / LOW
    band68: tuple[float, float]   # price levels
    band95: tuple[float, float]
    ret_q68: tuple[float, float]  # log-return quantiles, percent
    ret_q95: tuple[float, float]
    last_close: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["params"] = asdict(self.params)
        return d


# --------------------------------------------------------------------------- #
# Likelihood, filter and simulator (shared by the scipy fallback and tests)
# --------------------------------------------------------------------------- #

def _std_t_quantile(q: float, nu: float) -> float:
    """Quantile of a Student-t rescaled to unit variance (as `arch` uses)."""
    return stats.t.ppf(q, nu) * math.sqrt((nu - 2) / nu)


def filter_variance(r: np.ndarray, p: GarchParams) -> np.ndarray:
    """sigma_t^2 for t = 0..n (the last entry is the one-step-ahead forecast)."""
    eps = r - p.mu
    s2 = np.empty(len(r) + 1)
    s2[0] = np.var(eps)  # backcast, same spirit as arch's default
    for t in range(len(r)):
        neg = 1.0 if eps[t] < 0 else 0.0
        s2[t + 1] = p.omega + (p.alpha + p.gamma * neg) * eps[t] ** 2 + p.beta * s2[t]
    return s2


def neg_loglik(theta: np.ndarray, r: np.ndarray) -> float:
    p = GarchParams(*theta)
    if (p.omega <= 0 or p.alpha < 0 or p.beta < 0 or p.alpha + p.gamma < 0
            or p.persistence >= 0.9999 or p.nu <= 2.05):
        return 1e10
    s2 = filter_variance(r, p)[:-1]
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e10
    z2 = (r - p.mu) ** 2 / s2
    nu = p.nu
    ll = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
          - 0.5 * np.log(np.pi * (nu - 2)) - 0.5 * np.log(s2)
          - (nu + 1) / 2 * np.log1p(z2 / (nu - 2)))
    return -float(np.sum(ll))


def simulate(n: int, p: GarchParams, seed: int = 0, burn: int = 500) -> np.ndarray:
    """Simulate n returns (percent) from a GJR-GARCH-t process."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(p.nu, size=n + burn) * math.sqrt((p.nu - 2) / p.nu)
    r = np.empty(n + burn)
    s2 = p.omega / (1 - p.persistence)
    eps_prev = 0.0
    for t in range(n + burn):
        if t:
            neg = 1.0 if eps_prev < 0 else 0.0
            s2 = p.omega + (p.alpha + p.gamma * neg) * eps_prev ** 2 + p.beta * s2
        eps_prev = math.sqrt(s2) * z[t]
        r[t] = p.mu + eps_prev
    return r[burn:]


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

def fit_arch(r_pct: np.ndarray) -> GarchParams:
    from arch import arch_model

    res = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, o=1, q=1,
                     dist="t", rescale=False).fit(disp="off", show_warning=False)
    if res.convergence_flag != 0:
        raise RuntimeError(f"arch did not converge (flag {res.convergence_flag})")
    pr = res.params
    return GarchParams(mu=pr["mu"], omega=pr["omega"], alpha=pr["alpha[1]"],
                       gamma=pr["gamma[1]"], beta=pr["beta[1]"], nu=pr["nu"])


def fit_scipy(r_pct: np.ndarray) -> GarchParams:
    var = np.var(r_pct)
    x0 = np.array([np.mean(r_pct), 0.05 * var, 0.05, 0.05, 0.88, 8.0])
    bounds = [(-5 * abs(np.std(r_pct)), 5 * abs(np.std(r_pct))), (1e-8, 10 * var),
              (0.0, 0.5), (-0.5, 0.5), (0.0, 0.999), (2.1, 100.0)]
    res = optimize.minimize(neg_loglik, x0, args=(r_pct,), method="L-BFGS-B", bounds=bounds)
    if not np.isfinite(res.fun) or res.fun >= 1e10:
        raise RuntimeError(f"scipy GARCH fit failed: {res.message}")
    return GarchParams(*res.x)


def fit(r_pct: np.ndarray) -> tuple[GarchParams, str]:
    """Fit with `arch`; fall back to scipy if arch is missing or fails."""
    try:
        return fit_arch(r_pct), "arch"
    except Exception as exc:
        log.warning("arch fit failed (%s); using scipy fallback", exc)
        return fit_scipy(r_pct), "scipy"


# --------------------------------------------------------------------------- #
# Forecast, bands, coverage
# --------------------------------------------------------------------------- #

def log_returns_pct(close: pd.Series) -> pd.Series:
    return (np.log(close).diff() * 100).dropna()


def regime_label(sigma: float, long_run: float, high: float = 1.3, low: float = 0.75) -> str:
    if not math.isfinite(long_run):
        return "HIGH"
    if sigma > high * long_run:
        return "HIGH"
    if sigma < low * long_run:
        return "LOW"
    return "NORMAL"


def forecast(close: pd.Series, window: int = 1000, regime_high: float = 1.3,
             regime_low: float = 0.75) -> GarchForecast:
    """Fit on the last `window` daily log returns; forecast the next session."""
    r = log_returns_pct(close).to_numpy()[-window:]
    params, engine = fit(r)
    sigma = math.sqrt(filter_variance(r, params)[-1]) if engine == "scipy" else _arch_sigma_next(r, params)
    q = lambda p_: params.mu + sigma * _std_t_quantile(p_, params.nu)
    ret68 = (q(0.16), q(0.84))
    ret95 = (q(0.025), q(0.975))
    last = float(close.iloc[-1])
    price = lambda x: last * math.exp(x / 100)
    lr = params.long_run_sigma
    return GarchForecast(
        params=params, engine=engine, n_obs=len(r), sigma_next=sigma, long_run_sigma=lr,
        regime=regime_label(sigma, lr, regime_high, regime_low),
        band68=(price(ret68[0]), price(ret68[1])), band95=(price(ret95[0]), price(ret95[1])),
        ret_q68=ret68, ret_q95=ret95, last_close=last,
    )


def _arch_sigma_next(r: np.ndarray, p: GarchParams) -> float:
    """One-step-ahead sigma from arch with the fitted parameters fixed."""
    from arch import arch_model

    am = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False)
    res = am.fix([p.mu, p.omega, p.alpha, p.gamma, p.beta, p.nu])
    return float(np.sqrt(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]))


@dataclass
class Coverage:
    n_days: int
    inside68: float   # share of days whose close fell inside the 68% band
    inside95: float
    start: str
    end: str


def band_coverage(close: pd.Series, window: int = 1000, test_days: int = 250) -> Coverage:
    """Out-of-sample band check.

    Parameters are fit only on returns *before* the test period; the variance
    filter then runs through the test period with those parameters fixed, so
    each day's band uses data up to the prior close only.
    """
    r_all = log_returns_pct(close)
    if len(r_all) < test_days + 300:
        test_days = max(50, len(r_all) // 4)
    train = r_all.iloc[:-test_days].to_numpy()[-window:]
    params, _ = fit(train)
    # Filter over train + test so the test period's sigma is warmed up.
    r = np.concatenate([train, r_all.iloc[-test_days:].to_numpy()])
    s = np.sqrt(filter_variance(r, params)[:-1])[-test_days:]
    test = r[-test_days:]
    z = (test - params.mu) / s
    q68, q95 = _std_t_quantile(0.84, params.nu), _std_t_quantile(0.975, params.nu)
    idx = r_all.index[-test_days:]
    return Coverage(n_days=test_days, inside68=float(np.mean(np.abs(z) <= q68)),
                    inside95=float(np.mean(np.abs(z) <= q95)),
                    start=str(idx[0].date()), end=str(idx[-1].date()))
