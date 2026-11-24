"""Data layer: daily k-lines and the pre-market quote, cached to disk.

Rules (DESIGN.md, "Data layer"):
- Every fetch caches to disk; a failed fetch falls back to the cache instead
  of crashing the run, and the run manifest records that the data is stale.
- Timestamps are UTC internally. Daily bars are indexed by their session date
  (a tz-naive New York calendar date), which is not a point in time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
# Daily bars from yfinance are final some minutes after the 16:00 close.
CLOSE_FINAL_AFTER = dtime(16, 20)


@dataclass
class SourceStatus:
    name: str
    ok: bool
    stale: bool = False
    detail: str = ""
    last_bar: str | None = None


@dataclass
class RunManifest:
    started_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sources: list[SourceStatus] = field(default_factory=list)

    def add(self, status: SourceStatus) -> None:
        self.sources.append(status)

    @property
    def failed(self) -> list[SourceStatus]:
        return [s for s in self.sources if not s.ok or s.stale]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"started_utc": self.started_utc,
                   "sources": [asdict(s) for s in self.sources]}
        path.write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------- #
# Daily prices
# --------------------------------------------------------------------------- #

def _cache_path(data_dir: str | Path, ticker: str) -> Path:
    safe = ticker.replace("^", "_")
    return Path(data_dir) / "prices" / f"{safe}.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output to OHLCV indexed by tz-naive session date."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(-1, axis=1) if df.columns.nlevels > 1 else df
    df = df[[c for c in OHLCV if c in df.columns]].copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(NY).tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["Close"])


def drop_incomplete_session(df: pd.DataFrame, now_utc: datetime | None = None) -> pd.DataFrame:
    """Drop today's bar while the New York session is still open (partial close)."""
    now_ny = (now_utc or datetime.now(timezone.utc)).astimezone(NY)
    today = pd.Timestamp(now_ny.date())
    if len(df) and df.index[-1] == today and now_ny.time() < CLOSE_FINAL_AFTER:
        return df.iloc[:-1]
    return df


def _cache_is_fresh(path: Path, now_utc: datetime) -> bool:
    """Fresh = written after the most recent final close (so no new bar can exist)."""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    now_ny = now_utc.astimezone(NY)
    last_final = datetime.combine(now_ny.date(), CLOSE_FINAL_AFTER, tzinfo=NY)
    if now_ny < last_final:
        last_final -= timedelta(days=1)
    return mtime > last_final.astimezone(timezone.utc)


def _download(ticker: str, years: int) -> pd.DataFrame:
    import yfinance as yf

    start = (datetime.now(timezone.utc) - timedelta(days=int(365.25 * years) + 5)).date()
    raw = yf.Ticker(ticker).history(start=start.isoformat(), interval="1d",
                                    auto_adjust=True, actions=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    return _normalize(raw)


def get_prices(ticker: str, data_dir: str | Path, years: int = 6,
               manifest: RunManifest | None = None, offline: bool = False,
               now_utc: datetime | None = None) -> pd.DataFrame:
    """Split-adjusted daily OHLCV for `ticker`, refreshed at most once per close."""
    now_utc = now_utc or datetime.now(timezone.utc)
    path = _cache_path(data_dir, ticker)
    status = SourceStatus(name=f"prices:{ticker}", ok=True)

    if not offline and _cache_is_fresh(path, now_utc):
        df = pd.read_parquet(path)
        status.detail = "cache (fresh)"
    else:
        try:
            if offline:
                raise RuntimeError("offline mode")
            df = _download(ticker, years)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            status.detail = "downloaded"
        except Exception as exc:  # network, rate limit, schema change
            (log.info if offline else log.warning)("price fetch failed for %s: %s", ticker, exc)
            if not path.exists():
                status.ok = False
                status.detail = f"failed, no cache: {exc}"
                if manifest:
                    manifest.add(status)
                raise
            df = pd.read_parquet(path)
            status.stale = True
            status.detail = f"fetch failed, using cache: {exc}"

    df = drop_incomplete_session(df, now_utc)
    status.last_bar = str(df.index[-1].date()) if len(df) else None
    if manifest:
        manifest.add(status)
    return df


# --------------------------------------------------------------------------- #
# Pre-market quote
# --------------------------------------------------------------------------- #

@dataclass
class PreMarketQuote:
    ticker: str
    price: float | None
    asof_utc: str | None
    source: str
    detail: str = ""


def get_premarket(ticker: str, manifest: RunManifest | None = None,
                  offline: bool = False, now_utc: datetime | None = None) -> PreMarketQuote:
    """Latest extended-hours trade before today's 09:30 ET open.

    Tries today's 1-minute pre-market bars first, then `info['preMarketPrice']`.
    Returns price=None (never raises) when nothing is available, e.g. before
    04:00 ET or on a holiday; the report then shows a warning.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    status = SourceStatus(name=f"premarket:{ticker}", ok=False)
    quote = PreMarketQuote(ticker, None, None, "none")
    if offline:
        status.detail = "offline mode"
    else:
        try:
            import yfinance as yf

            t = yf.Ticker(ticker)
            bars = t.history(period="5d", interval="1m", prepost=True)
            if bars is not None and not bars.empty:
                idx = bars.index.tz_convert(NY)
                today = now_utc.astimezone(NY).date()
                mask = (idx.date == today) & (idx.time < dtime(9, 30))
                pre = bars[mask]
                if not pre.empty:
                    ts = pre.index[-1].tz_convert(timezone.utc)
                    quote = PreMarketQuote(ticker, float(pre["Close"].iloc[-1]),
                                           ts.isoformat(), "yfinance 1m prepost")
            if quote.price is None:
                info = t.info or {}
                p, ts = info.get("preMarketPrice"), info.get("preMarketTime")
                if p and ts:
                    ts_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    if ts_dt.astimezone(NY).date() == now_utc.astimezone(NY).date():
                        quote = PreMarketQuote(ticker, float(p), ts_dt.isoformat(),
                                               "yfinance info")
            if quote.price is None:
                status.detail = "no pre-market trades yet today"
            else:
                status.ok = True
                status.detail = quote.source
        except Exception as exc:
            log.warning("pre-market fetch failed for %s: %s", ticker, exc)
            status.detail = f"failed: {exc}"
    quote.detail = status.detail
    if manifest:
        manifest.add(status)
    return quote
