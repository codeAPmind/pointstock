"""Forecast ledger and daily scoring (DESIGN.md, "Forecast ledger" and "Scoring").

data/forecast_ledger.csv holds one row per ticker per forecast session. The
morning run first fills outcome columns for any earlier row whose session has
closed, then scores the last 60 scored rows, then appends today's forecast.
Re-running on the same day replaces that day's row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "date", "ticker", "last_close_date", "last_close", "premarket", "gap",
    "p_uncond", "p_prior", "p_ml", "p_news", "p_baserate", "p_final",
    "w_ml", "w_news", "w_baserate", "sigma_pred", "band68_low", "band68_high",
    "band95_low", "band95_high", "active_conditions", "event_types", "news_scorer",
    "outcome_up", "actual_close", "actual_ret", "inside_band68",
    "model_version", "prompt_version", "created_utc",
    "p_macro", "w_macro", "macro_events", "macro_vol_mult",
]
OUTCOME_COLS = ["outcome_up", "actual_close", "actual_ret", "inside_band68"]
SIGNALS = ["p_final", "p_ml", "p_news", "p_macro", "p_baserate", "p_prior"]
WINDOW = 60


def ledger_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "forecast_ledger.csv"


def load(data_dir: str | Path) -> pd.DataFrame:
    p = ledger_path(data_dir)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(p, dtype={"date": str, "last_close_date": str, "ticker": str})
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLUMNS]


def save(df: pd.DataFrame, data_dir: str | Path) -> Path:
    p = ledger_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["date", "ticker"]).to_csv(p, index=False)
    return p


def upsert(df: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Replace any existing (date, ticker) rows with the new forecasts."""
    new = pd.DataFrame(rows, columns=COLUMNS)
    keys = set(zip(new["date"], new["ticker"]))
    keep = df[[(d, t) not in keys for d, t in zip(df["date"], df["ticker"])]]
    out = pd.concat([keep, new], ignore_index=True) if len(keep) else new
    return out[COLUMNS]


def fill_outcomes(df: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """Fill outcomes for rows whose session close is now known. Returns (ledger, rows filled).

    Uses the current (split-adjusted) price series for both the reference
    close and the outcome close, so a later split does not fake a move.
    """
    df = df.copy()
    filled = 0
    for i, r in df[df["outcome_up"].isna()].iterrows():
        px = prices.get(r["ticker"])
        if px is None:
            continue
        d, prev = pd.Timestamp(r["date"]), pd.Timestamp(r["last_close_date"])
        if d not in px.index or prev not in px.index:
            continue
        close, ref = float(px.at[d, "Close"]), float(px.at[prev, "Close"])
        scale = ref / float(r["last_close"]) if r["last_close"] else 1.0   # split adjustment, normally 1
        lo, hi = float(r["band68_low"]) * scale, float(r["band68_high"]) * scale
        df.at[i, "actual_close"] = close
        df.at[i, "actual_ret"] = math.log(close / ref)
        df.at[i, "outcome_up"] = float(close > ref)
        df.at[i, "inside_band68"] = float(lo <= close <= hi) if not (math.isnan(lo) or math.isnan(hi)) else np.nan
        filled += 1
    return df, filled


@dataclass
class Scorecard:
    n_scored: int
    window: int
    bss: dict[str, float | None]          # signal -> BSS vs always-unconditional-rate
    n_by_signal: dict[str, int]
    band_coverage: float | None
    calibration: list[dict] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)   # signals whose weight is cut to zero

    def summary(self) -> str:
        if self.n_scored == 0:
            return "Ledger: no scored forecasts yet (outcomes fill in the morning after each session)."
        parts = [f"Last {min(self.n_scored, self.window)} scored rows:"]
        for k in ("p_final", "p_ml", "p_news", "p_macro", "p_baserate"):
            v = self.bss.get(k)
            if v is not None:
                parts.append(f"{k} BSS {v:+.3f} (n={self.n_by_signal[k]})")
        if self.band_coverage is not None:
            parts.append(f"band coverage {self.band_coverage:.0%}")
        if self.n_scored < self.window:
            parts.append(f"· {self.window - self.n_scored} more rows before weights are learned from the ledger")
        return " ".join(parts)


def score(df: pd.DataFrame, window: int = WINDOW) -> Scorecard:
    done = df[df["outcome_up"].notna()].sort_values("date")
    tail = done.groupby("ticker", group_keys=False).tail(window) if len(done) else done
    y = tail["outcome_up"].astype(float)
    base = tail["p_uncond"].astype(float)
    bss, ns = {}, {}
    for s in SIGNALS:
        m = tail[s].notna()
        ns[s] = int(m.sum())
        if m.sum() == 0:
            bss[s] = None
            continue
        b_base = float(np.mean((base[m] - y[m]) ** 2))
        b_sig = float(np.mean((tail.loc[m, s].astype(float) - y[m]) ** 2))
        bss[s] = 1 - b_sig / b_base if b_base > 0 else None
    cov = tail["inside_band68"].dropna()
    cal = []
    if len(tail):
        bins = np.arange(0.35, 0.70, 0.05)
        cut = pd.cut(tail["p_final"].astype(float), bins)
        for interval, g in tail.groupby(cut, observed=True):
            cal.append({"bin": f"{interval.left:.0%}-{interval.right:.0%}", "n": len(g),
                        "forecast": float(g["p_final"].mean()), "actual": float(g["outcome_up"].mean())})
    disabled = [s for s in ("p_ml", "p_news", "p_macro", "p_baserate")
                if ns[s] >= window and bss[s] is not None and bss[s] < 0]
    return Scorecard(n_scored=len(done), window=window, bss=bss, n_by_signal=ns,
                     band_coverage=float(cov.mean()) if len(cov) else None,
                     calibration=cal, disabled=disabled)
