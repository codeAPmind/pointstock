"""Base rates: "the last N times this stock was in today's situation, what happened next?"

Three kinds (DESIGN.md, "Base rates"):
1. Conditional base rates from the stock's own history, checked at the last close.
2. Gap base rates from the pre-market price vs the last close.
3. Corporate event priors from the curated `event_base_rates.json` (display only;
   detection runs on headlines once the news module lands in M4).

No look-ahead: a past day only counts as a case if its outcome was already
known at the last close (next-day outcome for 1-day stats, five days for 5-day).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

MIN_CASES = 25          # below this a rate is shown as context only
GAP_MIN = 0.01          # |gap| >= 1% triggers gap base rates
GAP_LO, GAP_HI = 0.6, 1.6


# --------------------------------------------------------------------------- #
# 1. Conditional base rates
# --------------------------------------------------------------------------- #

def condition_frame(px: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Boolean column per condition, evaluated at each day's close (causal)."""
    c, h = px["Close"], px["High"]
    r = feats["ret_1d"]
    simple = c.pct_change()
    sigma60 = r.rolling(60).std().shift(1)          # sigma *before* the day itself
    dd60 = c / h.rolling(60).max() - 1
    rsi, ma200 = feats["rsi14"], feats["dist_ma200"]
    cond = {
        "big_up_day": r > 2 * sigma60,
        "big_down_day": r < -2 * sigma60,
        "extreme_up_day": simple > 0.15,
        "extreme_down_day": simple < -0.15,
        "up_streak_3plus": feats["streak"] >= 3,
        "down_streak_3plus": feats["streak"] <= -3,
        "overbought_rsi70": rsi > 70,
        "oversold_rsi30": rsi < 30,
        "above_ma200": ma200 > 0,
        "below_ma200": ma200 < 0,
        "volume_spike_z2": feats["vol_z20"] > 2,
        "close_top10pct": feats["close_pos"] >= 0.9,
        "close_bottom10pct": feats["close_pos"] <= 0.1,
        "drawdown_30pct_60d": dd60 < -0.30,
    }
    out = pd.DataFrame(cond, index=px.index)
    # A condition that could not be evaluated (warm-up NaNs) is False.
    return out.fillna(False).astype(bool)


LABELS = {
    "big_up_day": "Big up day (> +2σ60)",
    "big_down_day": "Big down day (< −2σ60)",
    "extreme_up_day": "Extreme up day (> +15%)",
    "extreme_down_day": "Extreme down day (< −15%)",
    "up_streak_3plus": "3+ up closes in a row",
    "down_streak_3plus": "3+ down closes in a row",
    "overbought_rsi70": "Overbought (RSI14 > 70)",
    "oversold_rsi30": "Oversold (RSI14 < 30)",
    "above_ma200": "Above 200-day MA",
    "below_ma200": "Below 200-day MA",
    "volume_spike_z2": "Volume spike (z > 2)",
    "close_top10pct": "Closed in top 10% of range",
    "close_bottom10pct": "Closed in bottom 10% of range",
    "drawdown_30pct_60d": "Down > 30% from 60-day high",
}


@dataclass
class ConditionalRate:
    key: str
    label: str
    n: int
    p_up_1d: float
    mean_ret_1d: float      # simple return
    median_ret_1d: float
    n_5d: int
    p_up_5d: float
    usable: bool            # n >= MIN_CASES


def forward_returns(px: pd.DataFrame) -> pd.DataFrame:
    c = px["Close"]
    return pd.DataFrame({"fwd_1d": c.shift(-1) / c - 1, "fwd_5d": c.shift(-5) / c - 1})


def unconditional_rate(px: pd.DataFrame) -> tuple[float, int]:
    """The prior p0: share of past days followed by a higher close."""
    f = forward_returns(px)["fwd_1d"].dropna()
    return float((f > 0).mean()), int(len(f))


def conditional_rates(px: pd.DataFrame, feats: pd.DataFrame,
                      asof: pd.Timestamp | None = None) -> list[ConditionalRate]:
    """Stats for every condition that is true on `asof` (default: the last bar).

    Only days strictly before `asof` with an outcome known by `asof` count.
    """
    asof = asof if asof is not None else px.index[-1]
    px, feats = px.loc[:asof], feats.loc[:asof]
    conds = condition_frame(px, feats)
    fwd = forward_returns(px)          # NaN where the outcome is after `asof`
    today = conds.iloc[-1]
    hist = conds.iloc[:-1]
    out = []
    for key in conds.columns:
        if not today[key]:
            continue
        mask = hist[key]
        f1 = fwd.loc[mask.index[mask], "fwd_1d"].dropna()
        f5 = fwd.loc[mask.index[mask], "fwd_5d"].dropna()
        n = len(f1)
        out.append(ConditionalRate(
            key=key, label=LABELS[key], n=n,
            p_up_1d=float((f1 > 0).mean()) if n else float("nan"),
            mean_ret_1d=float(f1.mean()) if n else float("nan"),
            median_ret_1d=float(f1.median()) if n else float("nan"),
            n_5d=len(f5), p_up_5d=float((f5 > 0).mean()) if len(f5) else float("nan"),
            usable=n >= MIN_CASES))
    return out


# --------------------------------------------------------------------------- #
# 2. Gap base rates
# --------------------------------------------------------------------------- #

@dataclass
class GapRate:
    gap: float               # today's pre-market gap, simple
    lo: float                # matched range of past gaps (same sign)
    hi: float
    n: int
    p_close_above_open: float
    p_gap_fills: float       # intraday price touched the prior close
    mean_open_to_close: float
    median_open_to_close: float
    usable: bool
    p_up_vs_prev: float = float("nan")   # P(close > prior close): the gap-day prior p0
    # Gap-conditioned band for today's close, in sigma-scaled returns (see below).
    band68: tuple[float, float] | None = None
    band95: tuple[float, float] | None = None


def past_gaps(px: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = px["Open"], px["High"], px["Low"], px["Close"]
    pc = c.shift(1)
    gap = o / pc - 1
    up = gap > 0
    fills = np.where(up, l <= pc, h >= pc)
    # Prior-day sigma (20d, strictly past) to rescale outcomes across regimes.
    sig = np.log(c).diff().rolling(20).std().shift(1)
    return pd.DataFrame({
        "gap": gap, "fills": fills.astype(bool), "oc": c / o - 1,
        "close_vs_prev": np.log(c / pc), "sigma_prev": sig,
    }, index=px.index).dropna()


def gap_rates(px: pd.DataFrame, gap: float, sigma_today: float | None = None) -> GapRate | None:
    """Base rates for today's gap; None when |gap| < 1%.

    `px` must end at the last completed close (the pre-market day is not in it).
    `sigma_today` (daily, as a fraction) turns the matched days' close-vs-prior-
    close outcomes, measured in their own sigma units, into a gap-day band.
    """
    if gap is None or abs(gap) < GAP_MIN:
        return None
    g = past_gaps(px)
    a, b = sorted((GAP_LO * gap, GAP_HI * gap))
    m = g[(g["gap"] >= a) & (g["gap"] <= b)]
    n = len(m)
    rate = GapRate(gap=gap, lo=a, hi=b, n=n,
                   p_close_above_open=float((m["oc"] > 0).mean()) if n else float("nan"),
                   p_gap_fills=float(m["fills"].mean()) if n else float("nan"),
                   mean_open_to_close=float(m["oc"].mean()) if n else float("nan"),
                   median_open_to_close=float(m["oc"].median()) if n else float("nan"),
                   usable=n >= MIN_CASES,
                   p_up_vs_prev=float((m["close_vs_prev"] > 0).mean()) if n else float("nan"))
    if rate.usable and sigma_today:
        z = (m["close_vs_prev"] / m["sigma_prev"]).to_numpy()
        q = np.quantile(z, [0.025, 0.16, 0.84, 0.975]) * sigma_today
        rate.band95, rate.band68 = (q[0], q[3]), (q[1], q[2])
    return rate


def widened_bands(last_close: float, garch68: tuple[float, float], garch95: tuple[float, float],
                  gr: GapRate | None) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Event-day bands: the union of the GARCH band and the gap-conditioned band."""
    if gr is None or gr.band68 is None:
        return None
    px = lambda x: last_close * float(np.exp(x))
    b68 = (min(garch68[0], px(gr.band68[0])), max(garch68[1], px(gr.band68[1])))
    b95 = (min(garch95[0], px(gr.band95[0])), max(garch95[1], px(gr.band95[1])))
    return b68, b95


# --------------------------------------------------------------------------- #
# 3. Corporate event priors
# --------------------------------------------------------------------------- #

@dataclass
class EventPrior:
    event_type: str
    question: str
    keywords: list[str]
    prior: float | None
    curated: bool
    cases: list[dict] = field(default_factory=list)            # counted, verified outcomes
    candidate_cases: list[dict] = field(default_factory=list)  # named in DESIGN.md, not yet scored
    notes: str = ""

    @property
    def usable(self) -> bool:
        """Only curated priors may ever feed a probability; placeholders are display-only."""
        return self.curated and self.prior is not None and len(self.cases) > 0


def load_event_priors(path: str | Path) -> dict[str, EventPrior]:
    raw = json.loads(Path(path).read_text())
    return {k: EventPrior(event_type=k, **{f: v for f, v in e.items()
                                           if f in EventPrior.__dataclass_fields__ and f != "event_type"})
            for k, e in raw["events"].items()}


def detect_events(headlines: list[str], priors: dict[str, EventPrior]) -> dict[str, list[str]]:
    """Keyword match headlines to event types. Returns {event_type: [matching headlines]}."""
    hits: dict[str, list[str]] = {}
    for key, ep in priors.items():
        pats = [re.compile(r"\b" + re.escape(k.lower()) + r"\b") for k in ep.keywords]
        for h in headlines:
            if any(p.search(h.lower()) for p in pats):
                hits.setdefault(key, []).append(h)
    return hits


def to_dict(obj) -> dict:
    return asdict(obj)
