"""Feature table: one row per trading day, no look-ahead.

Every feature on row t uses only data known at the close of day t. The only
forward-looking columns are the targets (`target_up`, `next_ret`), which are
kept apart in TARGET_COLS and must never be fed to a model as inputs.
News features (DESIGN.md) arrive with the news module in M4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_COLS = ["target_up", "next_ret"]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def _streak(close: pd.Series) -> pd.Series:
    """Signed run length of consecutive up (+) or down (-) closes."""
    sign = np.sign(close.diff()).fillna(0).to_numpy()
    out = np.zeros(len(sign))
    for i in range(1, len(sign)):
        if sign[i] == 0:
            out[i] = 0
        elif sign[i] == np.sign(out[i - 1]):
            out[i] = out[i - 1] + sign[i]
        else:
            out[i] = sign[i]
    return pd.Series(out, index=close.index)


def stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price/volume features from one stock's OHLCV."""
    o, h, l, c, v = (df[k] for k in ["Open", "High", "Low", "Close", "Volume"])
    logc = np.log(c)
    r1 = logc.diff()
    f = pd.DataFrame(index=df.index)

    # Returns
    f["ret_1d"] = r1
    f["ret_5d"] = logc.diff(5)
    f["ret_20d"] = logc.diff(20)
    f["gap_overnight"] = np.log(o / c.shift(1))
    f["ret_open_close"] = np.log(c / o)

    # Trend
    for n in (20, 50, 200):
        f[f"dist_ma{n}"] = c / c.rolling(n).mean() - 1
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist_pct"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    # Mean reversion
    f["rsi14"] = _rsi(c)
    f["boll_z20"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    f["streak"] = _streak(c)

    # Volatility
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    f["atr14_pct"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c
    f["rvol_5"] = r1.rolling(5).std() * np.sqrt(252)
    f["rvol_20"] = r1.rolling(20).std() * np.sqrt(252)
    f["rvol_ratio"] = f["rvol_5"] / f["rvol_20"]
    f["range_pct"] = (h - l) / c

    # Volume
    lv = np.log(v.replace(0, np.nan))
    f["vol_z20"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()

    # Candle shape: 0 = closed at the low, 1 = closed at the high
    f["close_pos"] = ((c - l) / (h - l)).where(h > l, 0.5)
    return f


def market_features(market: dict[str, pd.DataFrame], sector: str | None = None) -> pd.DataFrame:
    """SPY/QQQ 1-day return, VIX level and 5-day change, optional sector ETF return."""
    cols = {}
    for name in ("SPY", "QQQ"):
        if name in market:
            cols[f"{name.lower()}_ret_1d"] = np.log(market[name]["Close"]).diff()
    if "VIX" in market:
        vix = market["VIX"]["Close"]
        cols["vix"] = vix
        cols["vix_chg_5d"] = vix - vix.shift(5)
    if sector and sector in market:
        cols["sector_ret_1d"] = np.log(market[sector]["Close"]).diff()
    return pd.DataFrame(cols)


def targets(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-looking labels. NaN on the last row (tomorrow is unknown)."""
    logc = np.log(df["Close"])
    nxt = logc.shift(-1) - logc
    t = pd.DataFrame(index=df.index)
    t["next_ret"] = nxt
    t["target_up"] = (nxt > 0).astype(float).where(nxt.notna())
    return t


def build_features(df: pd.DataFrame, market: dict[str, pd.DataFrame] | None = None,
                   sector: str | None = None) -> pd.DataFrame:
    """Full table = stock features + market context (aligned on the stock's dates) + targets."""
    feats = stock_features(df)
    if market:
        # Reindex onto the stock's calendar; forward-fill only from the past.
        mkt = market_features(market, sector).reindex(feats.index).ffill(limit=3)
        feats = feats.join(mkt)
    return feats.join(targets(df))


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in TARGET_COLS]


@dataclass(frozen=True)
class TechnicalRead:
    """Last-close snapshot of the moving averages, RSI and MACD already in the feature table.

    A reading aid for the report. It does not enter the combiner.
    """
    asof: str
    close: float
    ma20: float | None
    ma50: float | None
    ma200: float | None
    dist_ma20: float | None
    dist_ma50: float | None
    dist_ma200: float | None
    rsi14: float | None
    macd_hist_pct: float | None
    stack: str
    checks_up: int
    checks_n: int
    rsi_band: str
    macd_side: str


def _cell(row: pd.Series, name: str) -> float | None:
    if name not in row.index or pd.isna(row[name]):
        return None
    return float(row[name])


def _ma_level(close: float, dist: float | None) -> float | None:
    if dist is None:
        return None
    return close / (1 + dist)


def _stack(ma20: float | None, ma50: float | None, ma200: float | None) -> str:
    levels = (ma20, ma50, ma200)
    if any(v is None for v in levels):
        return "incomplete"
    if ma20 > ma50 > ma200:
        return "20 > 50 > 200"
    if ma20 < ma50 < ma200:
        return "20 < 50 < 200"
    return "mixed"


def _rsi_band(rsi: float | None) -> str:
    if rsi is None:
        return "n/a"
    if rsi < 30:
        return "below 30"
    if rsi < 50:
        return "30–50"
    if rsi <= 70:
        return "50–70"
    return "above 70"


def technical_read(table: pd.DataFrame, close: float, asof: str) -> TechnicalRead:
    """Read MA20/50/200, RSI(14) and MACD histogram off the last feature row."""
    row = table.iloc[-1]
    d20, d50, d200 = _cell(row, "dist_ma20"), _cell(row, "dist_ma50"), _cell(row, "dist_ma200")
    ma20, ma50, ma200 = _ma_level(close, d20), _ma_level(close, d50), _ma_level(close, d200)
    rsi, macd = _cell(row, "rsi14"), _cell(row, "macd_hist_pct")
    flags = []
    for dist in (d20, d50, d200):
        if dist is not None:
            flags.append(dist > 0)
    if rsi is not None:
        flags.append(rsi >= 50)
    if macd is not None:
        flags.append(macd > 0)
    return TechnicalRead(
        asof=asof, close=close, ma20=ma20, ma50=ma50, ma200=ma200,
        dist_ma20=d20, dist_ma50=d50, dist_ma200=d200, rsi14=rsi, macd_hist_pct=macd,
        stack=_stack(ma20, ma50, ma200), checks_up=sum(flags), checks_n=len(flags),
        rsi_band=_rsi_band(rsi),
        macd_side="n/a" if macd is None else ("above zero" if macd > 0 else "below zero"),
    )
