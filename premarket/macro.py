"""Macro and industry context for the pre-market report.

Company headlines miss what moves most stocks on most days: rates, inflation
and jobs data, the Fed, tariffs, and news about the stock's industry. This
module adds four pieces, following the findings of MacroEventQuntify
(docs/research_plan.md there):

1. Market pre-open: index, rate, oil, gold and dollar futures since the last
   close, plus the stock's gap split into market, sector and stock-specific
   parts using daily betas.
2. Macro calendar today: CPI / NFP / PPI / PCE / retail / GDP / claims from
   the FMP economic calendar (FMP_API_KEY), FOMC days, and Treasury Note/Bond
   auctions (fiscaldata, no key). When a release is already out, its surprise
   z = (actual - estimate) / sigma, signed so hawkish (rates up) is positive.
3. Event-day volatility: how much bigger this stock's moves have been on past
   releases of the same type (variance ratio, from release dates in
   MacroEventQuntify's events table). Widens the band when N >= 20.
4. Macro and industry headlines scored for this stock by the LLM, giving a
   small-weight signal (p_macro) that the ledger scores like the others.

What it deliberately does not do: turn a macro surprise into a direction
call. MacroEventQuntify's walk-forward found surprises rank the opening gap
(Rank IC ~0.05) but not returns after the open (IC ~0.005), so surprises
widen bands and are shown as context; they do not move P(up).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import llm as LLM
from .news import ScoredHeadline, dedupe

log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
UA = "Mozilla/5.0 (premarket-report)"
MACRO_WEIGHT = 0.2
MACRO_LOGIT_SCALE = 0.3
MIN_EVENT_N = 20
MAX_VOL_MULT = 2.0
PROMPT_VERSION = "macro-v1"

# FMP event names -> event key, the sign that makes "hawkish / rates up" positive,
# and a prior sigma of (actual - estimate) in the series' units.
FMP_EVENTS = [
    (r"^Core Inflation Rate MoM", "cpi", 1, 0.1),
    (r"^Inflation Rate MoM", "cpi", 1, 0.1),
    (r"^Non Farm Payrolls", "nfp", 1, 75.0),
    (r"^Unemployment Rate", "nfp", -1, 0.1),
    (r"^Average Hourly Earnings MoM", "nfp", 1, 0.1),
    (r"^Core PPI MoM", "ppi", 1, 0.2),
    (r"^PPI MoM", "ppi", 1, 0.2),
    (r"^Core PCE Price Index MoM", "pce", 1, 0.1),
    (r"^PCE Price Index MoM", "pce", 1, 0.1),
    (r"^Retail Sales MoM", "retail", 1, 0.4),
    (r"^Retail Sales Ex Autos MoM", "retail", 1, 0.4),
    (r"^GDP Growth Rate QoQ", "gdp", 1, 0.5),
    (r"^Initial Jobless Claims", "claims", -1, 15.0),
    (r"^ISM Manufacturing PMI", "ism", 1, 1.5),
    (r"^ISM Services PMI", "ism", 1, 1.5),
    (r"^Fed Interest Rate Decision", "fomc", 1, 0.25),
]
EVENT_NAMES = {"cpi": "CPI", "nfp": "Jobs report", "ppi": "PPI", "pce": "PCE inflation",
               "retail": "Retail sales", "gdp": "GDP", "claims": "Jobless claims", "ism": "ISM PMI",
               "fomc": "FOMC decision"}


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #

def _get_json(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# --------------------------------------------------------------------------- #
# 1. Market pre-open
# --------------------------------------------------------------------------- #

@dataclass
class FutureMove:
    name: str
    symbol: str
    ref: float | None        # price at the last regular close (16:00 ET)
    last: float | None
    change: float | None     # simple return since that close
    asof_utc: str | None


def _ref_and_last(bars: pd.DataFrame, ref_time: datetime, now: datetime) -> tuple[float | None, float | None, str | None]:
    if bars is None or bars.empty:
        return None, None, None
    idx = bars.index.tz_convert(timezone.utc)
    close = bars["Close"].to_numpy()
    before_ref = np.where(idx <= ref_time)[0]
    upto_now = np.where(idx <= now)[0]
    if not len(before_ref) or not len(upto_now):
        return None, None, None
    i, j = before_ref[-1], upto_now[-1]
    return float(close[i]), float(close[j]), idx[j].isoformat()


def futures_moves(futures: dict[str, str], prev_close_utc: datetime, now: datetime,
                  offline: bool = False) -> tuple[list[FutureMove], list[str]]:
    out, failed = [], []
    for name, sym in futures.items():
        if offline:
            out.append(FutureMove(name, sym, None, None, None, None))
            continue
        try:
            import yfinance as yf

            bars = yf.Ticker(sym).history(period="5d", interval="5m", prepost=True)
            ref, last, ts = _ref_and_last(bars, prev_close_utc, now)
            chg = (last / ref - 1) if ref and last else None
            out.append(FutureMove(name, sym, ref, last, chg, ts))
            if chg is None:
                failed.append(sym)
        except Exception as exc:
            log.warning("futures %s failed: %s", sym, exc)
            out.append(FutureMove(name, sym, None, None, None, None))
            failed.append(sym)
    return out, failed


@dataclass
class GapSplit:
    gap: float               # stock log gap
    beta_mkt: float
    beta_sector: float | None
    market: float            # log-gap contribution
    sector: float
    stock_specific: float
    r2: float
    n: int


def gap_decomposition(stock: pd.Series, spy: pd.Series, sector: pd.Series | None,
                      g_stock: float, g_spy: float, g_sector: float | None,
                      window: int = 250) -> GapSplit | None:
    """Split a pre-market log gap into market, sector and stock-specific parts.

    Betas come from daily log returns over the last `window` common days:
        r = a + b_m * r_spy + b_s * e_sector,  e_sector = r_sector - beta(sector, spy) * r_spy
    so the sector part only counts what the sector did beyond the market.
    """
    r = np.log(pd.concat({"s": stock, "m": spy, **({"x": sector} if sector is not None else {})},
                         axis=1).dropna()).diff().dropna().tail(window)
    if len(r) < 60:
        return None
    use_sector = sector is not None and g_sector is not None
    if use_sector:
        bx = np.polyfit(r["m"], r["x"], 1)[0]
        e = r["x"] - bx * r["m"]
        X = np.column_stack([np.ones(len(r)), r["m"], e])
    else:
        X = np.column_stack([np.ones(len(r)), r["m"]])
    coef, *_ = np.linalg.lstsq(X, r["s"].to_numpy(), rcond=None)
    fitted = X @ coef
    ss_res = float(((r["s"] - fitted) ** 2).sum())
    ss_tot = float(((r["s"] - r["s"].mean()) ** 2).sum())
    b_m = float(coef[1])
    market = b_m * g_spy
    if use_sector:
        b_s = float(coef[2])
        sector_part = b_s * (g_sector - bx * g_spy)
    else:
        b_s, sector_part = None, 0.0
    return GapSplit(gap=g_stock, beta_mkt=b_m, beta_sector=b_s, market=market, sector=sector_part,
                    stock_specific=g_stock - market - sector_part,
                    r2=1 - ss_res / ss_tot if ss_tot > 0 else 0.0, n=len(r))


# --------------------------------------------------------------------------- #
# 2. Macro calendar
# --------------------------------------------------------------------------- #

@dataclass
class MacroEvent:
    key: str                 # cpi, nfp, fomc, auction_10y, ...
    name: str
    time_et: str             # "08:30"
    source: str
    estimate: float | None = None
    actual: float | None = None
    previous: float | None = None
    unit: str = ""
    surprise_z: float | None = None   # hawkish-positive; None until released
    impact: str = ""


def classify_fmp(name: str) -> tuple[str, int, float] | None:
    for pat, key, sign, sigma in FMP_EVENTS:
        if re.search(pat, name):
            return key, sign, sigma
    return None


def parse_fmp_calendar(rows: list[dict], day: str) -> list[MacroEvent]:
    """US events on `day` (ET) that map to a tracked type. FMP timestamps are UTC."""
    out = []
    for r in rows:
        if r.get("country") != "US":
            continue
        c = classify_fmp(str(r.get("event", "")))
        if c is None:
            continue
        key, sign, sigma = c
        ts = pd.Timestamp(r["date"]).tz_localize("UTC").tz_convert(NY)
        if ts.date().isoformat() != day:
            continue
        num = lambda v: None if v in (None, "") or (isinstance(v, float) and math.isnan(v)) else float(v)
        est, act = num(r.get("estimate")), num(r.get("actual"))
        z = None
        if est is not None and act is not None:
            z = float(np.clip(sign * (act - est) / sigma, -4, 4))
        out.append(MacroEvent(key=key, name=str(r["event"]), time_et=ts.strftime("%H:%M"), source="FMP",
                              estimate=est, actual=act, previous=num(r.get("previous")),
                              unit=str(r.get("unit") or ""), surprise_z=z, impact=str(r.get("impact") or "")))
    return out


def fmp_calendar(day: str, key: str) -> list[MacroEvent]:
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    url = ("https://financialmodelingprep.com/stable/economic-calendar?"
           f"from={day}&to={nxt}&apikey={urllib.parse.quote(key)}")
    rows = _get_json(url)
    if not isinstance(rows, list):
        raise RuntimeError(f"FMP returned {str(rows)[:120]}")
    return parse_fmp_calendar(rows, day)


def auction_key(term: str) -> str | None:
    """'10-Year' -> auction_10y; reopenings like '19-Year 11-Month' round up to auction_20y."""
    m = re.match(r"(\d+)-Year(?:\s+(\d+)-Month)?", term or "")
    if not m:
        return None
    years = int(m.group(1)) + (1 if m.group(2) and int(m.group(2)) > 0 else 0)
    return f"auction_{years}y" if years in (2, 3, 5, 7, 10, 20, 30) else None


def treasury_auctions(day: str) -> list[MacroEvent]:
    url = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/"
           f"upcoming_auctions?filter=auction_date:eq:{day}&page%5Bsize%5D=100")
    rows = _get_json(url).get("data", [])
    out, seen = [], set()
    for r in rows:
        if r.get("security_type") not in ("Note", "Bond"):
            continue
        k = auction_key(r.get("security_term", ""))
        if k and k not in seen:
            seen.add(k)
            out.append(MacroEvent(key=k, name=f"{r['security_term']} {r['security_type']} auction",
                                  time_et="13:00", source="Treasury"))
    return out


def fomc_event(day: str, fomc_dates: list[str]) -> list[MacroEvent]:
    return [MacroEvent(key="fomc", name="FOMC rate decision", time_et="14:00", source="calendar")] \
        if day in set(fomc_dates) else []


def todays_calendar(day: str, cfg: dict, data_dir: str | Path, offline: bool = False,
                    env=None) -> tuple[list[MacroEvent], list[str]]:
    """All tracked macro events for `day`, cached per day; FMP refreshed each run (actuals arrive)."""
    env = env if env is not None else os.environ
    warnings: list[str] = []
    cache = Path(data_dir) / "macro" / f"{day}_calendar.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    events: list[MacroEvent] = []
    key = env.get("FMP_API_KEY")
    if offline:
        if cache.exists():
            events = [MacroEvent(**e) for e in json.loads(cache.read_text())]
            warnings.append("macro calendar from cache (offline)")
        return events, warnings
    if key:
        try:
            events += fmp_calendar(day, key)
        except Exception as exc:
            warnings.append(f"FMP economic calendar failed ({type(exc).__name__}); only FOMC and auctions checked")
    else:
        warnings.append("no FMP_API_KEY: CPI/jobs/PPI release days not detected (FOMC and auctions only)")
    if not any(e.key == "fomc" for e in events):
        events += fomc_event(day, cfg.get("fomc_dates", []))
    try:
        events += treasury_auctions(day)
    except Exception as exc:
        warnings.append(f"Treasury auction schedule failed ({type(exc).__name__})")
    cache.write_text(json.dumps([asdict(e) for e in events], indent=1))
    return events, warnings


def group_surprise(events: list[MacroEvent]) -> dict[str, float]:
    """Average released surprise per event key (e.g. headline + core CPI)."""
    zs: dict[str, list[float]] = {}
    for e in events:
        if e.surprise_z is not None:
            zs.setdefault(e.key, []).append(e.surprise_z)
    return {k: float(np.mean(v)) for k, v in zs.items()}


# --------------------------------------------------------------------------- #
# 3. Event-day volatility and historical sensitivity
# --------------------------------------------------------------------------- #

def load_release_dates(path: str | Path | None) -> dict[str, list[pd.Timestamp]]:
    if not path or not Path(path).exists():
        return {}
    df = pd.read_parquet(path, columns=["event_key", "date"])
    return {k: sorted(pd.to_datetime(g["date"]).dt.normalize().unique()) for k, g in df.groupby("event_key")}


def event_vol_ratio(close: pd.Series, dates: list[pd.Timestamp], asof: pd.Timestamp) -> tuple[float | None, int]:
    """Variance of close-to-close log returns on past event days / all days (only dates before `asof`)."""
    r = np.log(close).diff().dropna()
    r = r[r.index < asof]
    ev = r[r.index.isin([d for d in dates if d < asof])]
    if len(ev) < 5 or r.var() == 0:
        return None, len(ev)
    return float(ev.var() / r.var()), len(ev)


def vol_multiplier(close: pd.Series, events: list[MacroEvent], release_dates: dict, asof: pd.Timestamp
                   ) -> tuple[float, list[dict]]:
    """Band multiplier for today: sqrt of the largest qualifying variance ratio, capped at 2."""
    details, mult = [], 1.0
    for key in sorted({e.key for e in events}):
        ratio, n = event_vol_ratio(close, release_dates.get(key, []), asof)
        ok = ratio is not None and n >= MIN_EVENT_N
        details.append({"key": key, "ratio": ratio, "n": n, "used": bool(ok and ratio > 1)})
        if ok and ratio > 1:
            mult = max(mult, min(MAX_VOL_MULT, math.sqrt(ratio)))
    return mult, details


def widen(last_close: float, band: tuple[float, float], mult: float) -> tuple[float, float]:
    lo, hi = band
    return (last_close * math.exp(mult * math.log(lo / last_close)),
            last_close * math.exp(mult * math.log(hi / last_close)))


def sensitivity(path: str | Path | None, ticker: str) -> dict[str, dict]:
    """Per-event reaction slope from MacroEventQuntify's pool_sens (return per 1-sigma hawkish shock)."""
    if not path or not Path(path).exists():
        return {}
    df = pd.read_parquet(path)
    df = df[df["ticker"] == ticker]
    return {r.event_key: {"b": float(r.b), "t": float(r.t_b), "n": int(r.n),
                          "b_hawk": float(r.b_hawk), "b_dove": float(r.b_dove)} for r in df.itertuples()}


# --------------------------------------------------------------------------- #
# 4. Macro and industry headlines
# --------------------------------------------------------------------------- #

def fetch_headlines(queries: list[str], cache_name: str, data_dir: str | Path, now: datetime,
                    hours: int = 24, offline: bool = False) -> tuple[list[dict], list[str]]:
    from .data import _google_news

    day = now.astimezone(NY).date().isoformat()
    path = Path(data_dir) / "news" / f"{day}_{cache_name}_raw.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    items, failed = [], []
    if offline and path.exists():
        items = json.loads(path.read_text())
        failed = ["offline"]
    elif not offline:
        for q in queries:
            try:
                got = _google_news(q)
                for h in got:
                    h["query"] = q
                items += got
            except Exception as exc:
                log.warning("macro headlines '%s' failed: %s", q, exc)
                failed.append(q)
            time.sleep(1.0)    # be gentle with Google News
        if failed and path.exists():
            items += [h for h in json.loads(path.read_text()) if h.get("query") in failed]
        if items:
            path.write_text(json.dumps(items, indent=1))
    cutoff = now - timedelta(hours=hours)
    recent = [h for h in dedupe(items) if pd.Timestamp(h["published_utc"]) >= pd.Timestamp(cutoff)]
    return recent, failed


MACRO_SYSTEM = """You score macro-economic and industry headlines for their likely effect on one stock today.

Stock: {company} ({ticker}). Main exposures: {exposure}.

For each numbered headline return:
- category: "macro" (economy, rates, Fed, inflation, jobs, trade policy, geopolitics, broad market), "industry" (the stock's sector, suppliers, customers, competitors, regulation of its industry), or "none" (irrelevant to this stock).
- sentiment: likely effect on {ticker}'s share price today, from -1 (clearly negative) to +1 (clearly positive); 0 if neutral, mixed or unclear. Judge through the exposures above (for example, higher yields usually hurt long-duration growth stocks and housing).
- materiality: 0 = irrelevant, 1 = background, 2 = notable, 3 = market-moving (a data surprise, a Fed decision, new tariffs or export controls, a major peer's guidance).
- why: at most 15 words naming the channel from the headline to {ticker}.

Judge only from the headline text. Headlines about {company} itself get category "none" (they are scored elsewhere)."""


def _macro_schema() -> dict:
    return {"type": "object", "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"i": {"type": "integer"}, "category": {"type": "string", "enum": ["macro", "industry", "none"]},
                       "sentiment": {"type": "number"}, "materiality": {"type": "integer"},
                       "why": {"type": "string"}},
        "required": ["i", "category", "sentiment", "materiality", "why"], "additionalProperties": False}}},
        "required": ["items"], "additionalProperties": False}


@dataclass
class MacroNews:
    headlines: list[ScoredHeadline]
    n_fetched: int
    scorer: str                     # "deepseek:<model>", "claude-haiku-5-5", or "unscored"
    score: float | None             # materiality-weighted mean sentiment of macro+industry items
    warnings: list[str] = field(default_factory=list)


def parse_macro_items(data: dict, items: list[dict]) -> list[ScoredHeadline]:
    by_i = {int(r["i"]): r for r in data.get("items", []) if "i" in r}
    out = []
    for i, h in enumerate(items, start=1):
        r = by_i.get(i)
        if r is None or r.get("category") not in ("macro", "industry"):
            continue
        out.append(ScoredHeadline(h["title"], h["source"], h["published_utc"], h.get("url"),
                                  float(np.clip(r.get("sentiment", 0), -1, 1)),
                                  int(np.clip(r.get("materiality", 0), 0, 3)),
                                  r["category"], str(r.get("why", ""))[:200]))
    return out


def score_macro_news(ticker: str, company: str, exposure: str, items: list[dict], provider: str | None,
                     max_items: int = 50) -> MacroNews:
    items = sorted(items, key=lambda h: h["published_utc"], reverse=True)[:max_items]
    if not items:
        return MacroNews([], 0, "unscored", None)
    if provider is None:
        return MacroNews([], len(items), "unscored", None,
                         ["no LLM key: macro/industry headlines not scored (weight 0)"])
    system = MACRO_SYSTEM.format(company=company, ticker=ticker, exposure=exposure)
    try:
        data, scorer, items = _score_with_retry(system, items, provider)
    except Exception as exc:
        log.warning("macro headline scoring failed for %s: %s", ticker, exc)
        return MacroNews([], len(items), "unscored", None,
                         [f"macro/industry headline scoring failed ({type(exc).__name__}); weight 0"])
    scored = parse_macro_items(data, items)
    w = np.array([h.materiality for h in scored], float)
    score = float((w * np.array([h.sentiment for h in scored])).sum() / w.sum()) if w.sum() > 0 else None
    return MacroNews(scored, len(items), scorer, score)


def _score_with_retry(system: str, items: list[dict], provider: str) -> tuple[dict, str, list[dict]]:
    """One call; if the reply is cut off for length, retry once with the newest half of the headlines."""
    try:
        data, scorer = _score_call(system, items, provider)
        return data, scorer, items
    except RuntimeError as exc:
        if "length" not in str(exc) or len(items) < 10:
            raise
        half = items[: len(items) // 2]
        log.info("macro scoring cut off at %d headlines; retrying with %d", len(items), len(half))
        data, scorer = _score_call(system, half, provider)
        return data, scorer, half


def _score_call(system: str, items: list[dict], provider: str) -> tuple[dict, str]:
    numbered = "\n".join(f"{i}. [{h['source']}] {h['title']}" for i, h in enumerate(items, start=1))
    user = f"Headlines:\n{numbered}\n\nScore every numbered headline; use its number as i."
    if provider == "deepseek":
        data, model = LLM.deepseek_json(system, user, _macro_schema(), role="news")
        return data, f"deepseek:{model}"
    import anthropic

    from .news import LLM_MODEL
    resp = anthropic.Anthropic().messages.create(
        model=LLM_MODEL, max_tokens=8000, system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": _macro_schema()}})
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"stopped with {resp.stop_reason}")
    return json.loads(next(b.text for b in resp.content if b.type == "text")), LLM_MODEL


def macro_probability(score: float | None, p_ref: float) -> float | None:
    if score is None:
        return None
    return 1 / (1 + math.exp(-(math.log(p_ref / (1 - p_ref)) + MACRO_LOGIT_SCALE * score)))


# --------------------------------------------------------------------------- #
# Per-ticker bundle for the report
# --------------------------------------------------------------------------- #

@dataclass
class MacroContext:
    futures: list[FutureMove]
    events: list[MacroEvent]
    surprises: dict[str, float]
    gap_split: GapSplit | None
    sector_symbol: str | None
    sector_gap: float | None
    vol_mult: float
    vol_details: list[dict]
    sensitivity: dict[str, dict]
    news: MacroNews | None
    warnings: list[str] = field(default_factory=list)
    prompt_version: str = PROMPT_VERSION
