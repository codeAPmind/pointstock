"""News tier 1: every headline, every day (DESIGN.md, "News and event analysis").

1. Relevance filter: keep headlines that name the company or ticker; drop
   list-style pieces ("10 stocks to watch").
2. Sentiment: one Claude call per ticker returns, per headline, sentiment
   (-1..+1), materiality (0-3), event_type (from event_base_rates.json) and a
   one-line why. Without credentials (or on any API failure) a keyword
   lexicon gives the same fields, and the report says which scorer ran.
3. Daily news score = materiality-weighted mean sentiment, plus a headline
   count z-score against the ticker's own recent history.
4. Event flags: event types matched against the curated priors table.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import llm as LLM
from .baserates import EventPrior, detect_events

log = logging.getLogger(__name__)

LLM_MODEL = "claude-haiku-5-5"
PROMPT_VERSION = "news-v1"
NEWS_LOGIT_SCALE = 0.4      # daily score +1 moves the logit by 0.4 (~ +10pp); refit from the ledger
MAX_HEADLINES = 40

LISTICLE = re.compile(
    r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(top\s+|best\s+|hot\s+)?"
    r"(ai\s+|tech\s+|growth\s+|chip\s+|ev\s+)?(stocks|picks|names|shares)\b"
    r"|stocks? to (watch|buy|sell)|top (gainers|losers|movers)|biggest (movers|gainers|losers)"
    r"|stocks making the biggest moves|premarket movers|market wrap|morning bid",
    re.I)


@dataclass
class ScoredHeadline:
    title: str
    source: str
    published_utc: str
    url: str | None
    sentiment: float          # -1..+1
    materiality: int          # 0..3
    event_type: str           # key in event_base_rates.json, or "none"
    why: str


@dataclass
class NewsResult:
    ticker: str
    scorer: str                         # "deepseek:<model>", "claude-haiku-5-5" or "lexicon"
    prompt_version: str
    headlines: list[ScoredHeadline]
    n_raw: int
    n_relevant: int
    score: float | None                 # materiality-weighted mean sentiment
    count_z: float | None
    events: dict[str, list[str]]        # event_type -> titles
    failed_feeds: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Relevance and de-duplication
# --------------------------------------------------------------------------- #

def _key(title: str) -> str:
    return re.sub(r"\s+", " ", title.lower()).strip()[:80]


def dedupe(items: list[dict]) -> list[dict]:
    """De-duplicate on the first 80 characters, lower-cased; keep the earliest copy."""
    seen, out = set(), []
    for h in sorted(items, key=lambda h: h["published_utc"]):
        k = _key(h["title"])
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out


# Tickers that are ordinary words: a bare match ("open door") is not evidence.
WORD_TICKERS = {"OPEN", "A", "IT", "ON", "ALL", "ARE", "CAN", "NOW", "BIG", "CAR", "LOVE", "FUN", "PLAY", "RUN"}


def is_relevant(title: str, ticker: str, names: list[str]) -> bool:
    if LISTICLE.search(title):
        return False
    t = re.escape(ticker)
    pats = [rf"\b{re.escape(n)}\b" for n in names]
    pats += [rf"\${t}\b", rf"\({t}\)", rf"(NASDAQ|NYSE):\s*{t}\b"]
    if ticker.upper() not in WORD_TICKERS:
        pats.append(rf"\b{t}\b")
    return any(re.search(p, title, flags=re.I) for p in pats)


def filter_relevant(items: list[dict], ticker: str, names: list[str]) -> list[dict]:
    return [h for h in dedupe(items) if is_relevant(h["title"], ticker, names)]


# --------------------------------------------------------------------------- #
# Scorers
# --------------------------------------------------------------------------- #

# Word stems (prefix match) for the no-API fallback scorer.
POS = ("beat", "surg", "soar", "jump", "rall", "record", "upgrad", "raise", "raising", "strong",
       "gain", "rise", "rising", "climb", "outperform", "bullish", "approv", "win", "boost",
       "reinstat", "rebound", "partner", "tops", "higher", "buyback", "profit", "growth", "euphori",
       "optimis", "upside", "breakthrough", "expand")
NEG = ("miss", "plung", "drop", "fall", "fell", "slump", "sink", "sank", "tumbl", "downgrad", "cut",
       "weak", "loss", "probe", "investigat", "lawsuit", "sue", "recall", "resign", "oust", "bearish",
       "lower", "warn", "delay", "fraud", "subpoena", "declin", "crash", "layoff", "delist", "worst",
       "slid", "scandal", "selloff", "sell-off", "concern", "fear", "downside", "halt")
NEGATORS = {"not", "no", "despite", "fails", "failed"}


def _polarity(title: str) -> tuple[int, int]:
    words = re.findall(r"[a-z][a-z'-]*", title.lower())
    p = sum(any(w.startswith(x) for x in POS) for w in words)
    n = sum(any(w.startswith(x) for x in NEG) for w in words)
    if any(w in NEGATORS for w in words):
        p, n = (p + n) // 2, (p + n) - (p + n) // 2   # negation: treat as mixed
    return p, n


HIGH_MATERIALITY = {"ceo_departure", "ceo_return", "merger_talks", "regulatory_legal"}


def score_lexicon(items: list[dict], priors: dict[str, EventPrior]) -> list[ScoredHeadline]:
    out = []
    for h in items:
        p, n = _polarity(h["title"])
        sent = 0.0 if p == n else 0.6 * (p - n) / (p + n)
        ev = detect_events([h["title"]], priors)
        etype = next((k for k in ev if k in HIGH_MATERIALITY), next(iter(ev), "none"))
        mat = 3 if etype in HIGH_MATERIALITY else 2 if etype != "none" else (1 if p or n else 0)
        out.append(ScoredHeadline(h["title"], h["source"], h["published_utc"], h.get("url"),
                                  sent, mat, etype, "keyword lexicon"))
    return out


SYSTEM_PROMPT = """You score news headlines for a pre-market stock report.

For each numbered headline about {company} ({ticker}), return:
- sentiment: the likely effect on {ticker}'s share price, from -1 (clearly negative) to +1 (clearly positive); 0 if neutral or unclear.
- materiality: 0 = not about the company or no price relevance, 1 = minor, 2 = notable (earnings, guidance, analyst action, product news), 3 = major event (CEO change, merger or takeover, regulatory or legal action, guidance shock).
- event_type: one of {event_types}, or "none".
- why: one short sentence citing what in the headline drives the score.

Judge only from the headline text; do not assume facts that are not stated. Headlines that only mention {ticker} in passing get materiality 0 or 1."""


def _schema(event_types: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "i": {"type": "integer"},
                "sentiment": {"type": "number"},
                "materiality": {"type": "integer"},
                "event_type": {"type": "string", "enum": event_types + ["none"]},
                "why": {"type": "string"},
            },
            "required": ["i", "sentiment", "materiality", "event_type", "why"],
            "additionalProperties": False}}},
        "required": ["items"],
        "additionalProperties": False,
    }


def parse_llm_items(text: str, items: list[dict], event_types: list[str]) -> list[ScoredHeadline]:
    """Turn the model's JSON into ScoredHeadlines; clip values; ignore unknown indices."""
    data = json.loads(text)
    by_i = {int(r["i"]): r for r in data.get("items", [])}
    out = []
    for i, h in enumerate(items, start=1):
        r = by_i.get(i)
        if r is None:
            continue
        et = r.get("event_type", "none")
        out.append(ScoredHeadline(
            h["title"], h["source"], h["published_utc"], h.get("url"),
            float(np.clip(r["sentiment"], -1, 1)), int(np.clip(r["materiality"], 0, 3)),
            et if et in event_types else "none", str(r.get("why", ""))[:200]))
    return out


def score_deepseek(items: list[dict], ticker: str, company: str,
                   event_types: list[str]) -> tuple[list[ScoredHeadline], str]:
    """One JSON-mode DeepSeek call for all of a ticker's headlines. Returns (scored, model)."""
    numbered = "\n".join(f"{i}. [{h['source']}] {h['title']}" for i, h in enumerate(items, start=1))
    data, model = LLM.deepseek_json(
        SYSTEM_PROMPT.format(company=company, ticker=ticker, event_types=", ".join(event_types)),
        f"Headlines:\n{numbered}\n\nScore every numbered headline; use its number as i.",
        _schema(event_types), role="news")
    return parse_llm_items(json.dumps(data), items, event_types), model


def score_llm(items: list[dict], ticker: str, company: str, event_types: list[str],
              client=None) -> list[ScoredHeadline]:
    """One structured-output Claude call for all of a ticker's headlines."""
    import anthropic

    client = client or anthropic.Anthropic()
    numbered = "\n".join(f"{i}. [{h['source']}] {h['title']}" for i, h in enumerate(items, start=1))
    resp = client.messages.create(
        model=LLM_MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT.format(company=company, ticker=ticker, event_types=", ".join(event_types)),
        messages=[{"role": "user", "content": f"Headlines:\n{numbered}"}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": _schema(event_types)}},
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"headline scoring stopped with {resp.stop_reason}")
    text = next(b.text for b in resp.content if b.type == "text")
    return parse_llm_items(text, items, event_types)


def llm_available() -> bool:
    return LLM.anthropic_available()


# --------------------------------------------------------------------------- #
# Daily score, count z-score, probability mapping
# --------------------------------------------------------------------------- #

def daily_score(scored: list[ScoredHeadline]) -> float | None:
    w = np.array([h.materiality for h in scored], float)
    if w.sum() <= 0:
        return None
    s = np.array([h.sentiment for h in scored])
    return float((w * s).sum() / w.sum())


def count_zscore(counts_path: Path, ticker: str, day: str, n: int, min_days: int = 10) -> float | None:
    """z-score of today's relevant-headline count vs the previous 20 recorded days; records today."""
    df = pd.read_csv(counts_path) if counts_path.exists() else pd.DataFrame(columns=["date", "ticker", "count"])
    df = df[~((df["date"] == day) & (df["ticker"] == ticker))]
    hist = df[df["ticker"] == ticker].sort_values("date")["count"].tail(20).astype(float)
    df = pd.concat([df, pd.DataFrame([{"date": day, "ticker": ticker, "count": n}])]).sort_values(["date", "ticker"])
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(counts_path, index=False)
    if len(hist) < min_days or hist.std() == 0:
        return None
    return float((n - hist.mean()) / hist.std())


def news_probability(score: float | None, p_ref: float) -> float | None:
    if score is None:
        return None
    lo = math.log(p_ref / (1 - p_ref)) + NEWS_LOGIT_SCALE * score
    return 1 / (1 + math.exp(-lo))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def analyze(ticker: str, company: str, names: list[str], raw: list[dict], failed_feeds: list[str],
            priors: dict[str, EventPrior], data_dir: str | Path, day: str,
            use_llm: bool | None = None, client=None, provider: str | None = "auto") -> NewsResult:
    """provider: "deepseek", "anthropic", "auto" (pick from the environment) or None (lexicon).
    A `client` (tests) forces the Claude path."""
    warnings = []
    rel = filter_relevant(raw, ticker, names)
    rel = sorted(rel, key=lambda h: h["published_utc"], reverse=True)[:MAX_HEADLINES]
    event_types = list(priors)
    scorer = "lexicon"
    scored: list[ScoredHeadline] = []
    if rel:
        if client is not None:
            prov = "anthropic"
        elif use_llm is False or provider is None:
            prov = None
        elif provider in ("deepseek", "anthropic"):
            prov = provider
        else:
            prov = LLM.active_provider(provider or "auto")
        if prov == "deepseek":
            try:
                scored, model = score_deepseek(rel, ticker, company, event_types)
                scorer = f"deepseek:{model}"
            except Exception as exc:
                log.warning("DeepSeek headline scoring failed for %s: %s", ticker, exc)
                warnings.append(f"DeepSeek headline scoring failed, used keyword lexicon ({exc})"[:200])
        elif prov == "anthropic":
            try:
                scored = score_llm(rel, ticker, company, event_types, client)
                scorer = LLM_MODEL
            except Exception as exc:
                log.warning("LLM headline scoring failed for %s: %s", ticker, exc)
                warnings.append(f"Claude headline scoring failed, used keyword lexicon ({type(exc).__name__})")
        else:
            warnings.append("no LLM key (DEEPSEEK_API_KEY): headlines scored by keyword lexicon")
        if not scored:
            scored = score_lexicon(rel, priors)
    events: dict[str, list[str]] = {}
    for h in scored:
        if h.event_type != "none" and h.materiality >= 2:
            events.setdefault(h.event_type, []).append(h.title)
    for et, titles in detect_events([h.title for h in scored], priors).items():
        events.setdefault(et, [])
        events[et] += [t for t in titles if t not in events[et]]
    d = Path(data_dir) / "news"
    res = NewsResult(ticker=ticker, scorer=scorer, prompt_version=PROMPT_VERSION, headlines=scored,
                     n_raw=len(raw), n_relevant=len(rel), score=daily_score(scored),
                     count_z=count_zscore(d / "counts.csv", ticker, day, len(rel)),
                     events=events, failed_feeds=failed_feeds, warnings=warnings)
    if failed_feeds:
        res.warnings.append(f"news source failed: {', '.join(failed_feeds)} (used cache where available)")
    (d / f"{day}_{ticker}.json").write_text(json.dumps(
        {**res.to_dict(), "model": scorer, "system_prompt": SYSTEM_PROMPT if scorer != "lexicon" else None,
         "generated_utc": datetime.now(timezone.utc).isoformat()}, indent=1))
    return res
