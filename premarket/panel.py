"""News tier 2: the analyst panel, only for materiality-3 events (DESIGN.md).

Steps, each a separate Claude call so the personas stay independent:
1. Actor map: key actors, what they want, what they can do, credibility, limits.
2. Scenario tree: 3-5 mutually exclusive outcomes with a time frame, multi-step
   paths broken into stages, a price estimate under each, and 3-5 tripwires.
3. Five personas (bull, bear, lawyer/regulator, the board's view, base-rate
   statistician) each give a probability per scenario. The statistician only
   sees the event table and the scenario list, never the headlines.
4. Aggregate in log-odds space (geometric mean of odds), renormalise, shrink
   toward the curated event prior when one exists; persona spread is the
   disagreement score.
5. Expected value = sum p x price, compared with the pre-market price.
6. Tripwires are checked against the next mornings' headlines; a hit re-runs
   the panel and appends to the event's dated history in data/events/.

Prompt rules: every claim cites a headline number from that day's input
(0 = the event table); uncited claims are dropped. Prompts and model are
stored with each run. The panel's weight in P(up) is 0 until 20 scored events.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from . import llm as LLM
from .baserates import EventPrior

log = logging.getLogger(__name__)

PANEL_MODEL = "claude-opus-5-5"
PANEL_EFFORT = "high"
PANEL_PROMPT_VERSION = "panel-v1"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
ACTIVE_DAYS = 30            # an event stays "live" (tripwires checked) this long
SHRINK = 0.5                # weight on the curated prior when one applies
PERSONAS = {
    "bull": "a bullish equity analyst who looks for the upside case, but only from the evidence",
    "bear": "a bearish short-seller who looks for the downside case, but only from the evidence",
    "lawyer": "a securities lawyer and former regulator focused on legal, governance and regulatory constraints",
    "board": "the company's board of directors, weighing fiduciary duty, control and reputation",
    "statistician": "a base-rate statistician who ignores the story and reasons only from how often outcomes like these happen",
}

COMMON_RULES = """Rules:
- Use only the numbered evidence given. Cite the evidence numbers that support each claim in "cites" (0 = the event table).
- Claims without a citation will be discarded.
- Do not invent facts, dates, prices or quotes."""


# --------------------------------------------------------------------------- #
# Data shapes
# --------------------------------------------------------------------------- #

@dataclass
class PanelRun:
    date: str
    trigger: str                    # "new" or "tripwire: <text>"
    model: str
    prompt_version: str
    evidence: list[str]
    actors: list[dict]
    scenarios: list[dict]           # id, label, timeframe, stages, price, yes_for_prior
    personas: dict[str, dict]       # name -> {probs: {id: p}, points: [...]}
    aggregate: dict[str, float]     # scenario id -> probability
    disagreement: float             # mean across scenarios of persona std-dev
    disagreement_label: str
    shrunk_to_prior: bool
    expected_price: float | None
    reference_price: float | None
    tripwires: list[dict]           # {signal, keywords}
    dropped_claims: int = 0


@dataclass
class EventRecord:
    event_id: str
    ticker: str
    event_type: str
    first_seen: str
    history: list[PanelRun] = field(default_factory=list)

    @property
    def latest(self) -> PanelRun | None:
        return self.history[-1] if self.history else None

    def to_dict(self) -> dict:
        return asdict(self)


def events_dir(data_dir: str | Path) -> Path:
    d = Path(data_dir) / "events"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_events(data_dir: str | Path, ticker: str) -> list[EventRecord]:
    out = []
    for p in sorted(events_dir(data_dir).glob(f"{ticker}_*.json")):
        raw = json.loads(p.read_text())
        rec = EventRecord(raw["event_id"], raw["ticker"], raw["event_type"], raw["first_seen"],
                          [PanelRun(**h) for h in raw["history"]])
        out.append(rec)
    return out


def save_event(rec: EventRecord, data_dir: str | Path) -> Path:
    p = events_dir(data_dir) / f"{rec.event_id}.json"
    p.write_text(json.dumps(rec.to_dict(), indent=1, default=str))
    return p


def active_event(records: list[EventRecord], event_type: str, today: str) -> EventRecord | None:
    """Most recent record of this type first seen within ACTIVE_DAYS of today."""
    live = [r for r in records if r.event_type == event_type
            and (date.fromisoformat(today) - date.fromisoformat(r.first_seen)).days <= ACTIVE_DAYS]
    return max(live, key=lambda r: r.first_seen) if live else None


def fired_tripwires(rec: EventRecord, headlines: list[dict]) -> list[str]:
    """Tripwire signals whose keywords appear in headlines newer than the last panel run."""
    last = rec.latest
    if last is None:
        return []
    newer = [h["title"].lower() for h in headlines if h["published_utc"][:10] > last.date
             or (h["published_utc"][:10] == last.date and h["title"] not in last.evidence)]
    hits = []
    for tw in last.tripwires:
        kws = [k.lower() for k in tw.get("keywords", []) if k.strip()]
        if kws and any(all(w in t for w in k.split()) for k in kws for t in newer):
            hits.append(tw["signal"])
    return hits


# --------------------------------------------------------------------------- #
# LLM calls
# --------------------------------------------------------------------------- #

def _call(client, system: str, user: str, schema: dict, max_tokens: int = 16000) -> dict:
    if client == "deepseek":
        data, _ = LLM.deepseek_json(system, user, schema, role="panel", max_tokens=8000)
        return data
    resp = client.beta.messages.create(
        model=PANEL_MODEL,
        max_tokens=max_tokens,
        betas=[FALLBACK_BETA],
        fallbacks="default",
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": PANEL_EFFORT, "format": {"type": "json_schema", "schema": schema}},
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"panel call stopped with {resp.stop_reason}")
    text = [b.text for b in resp.content if getattr(b, "type", "") == "text"][-1]
    return json.loads(text)


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or list(props),
            "additionalProperties": False}


CITES = {"type": "array", "items": {"type": "integer"}}
ACTORS_SCHEMA = _obj({"actors": {"type": "array", "items": _obj({
    "name": {"type": "string"}, "wants": {"type": "string"}, "can_do": {"type": "string"},
    "credibility": {"type": "string"}, "constraints": {"type": "string"}, "cites": CITES})}})
SCENARIO_SCHEMA = _obj({
    "scenarios": {"type": "array", "items": _obj({
        "id": {"type": "string"}, "label": {"type": "string"}, "timeframe": {"type": "string"},
        "stages": {"type": "array", "items": {"type": "string"}},
        "price": {"type": "number"}, "yes_for_prior": {"type": "boolean"}, "cites": CITES})},
    "tripwires": {"type": "array", "items": _obj({
        "signal": {"type": "string"}, "keywords": {"type": "array", "items": {"type": "string"}}})}})
PERSONA_SCHEMA = _obj({
    "probabilities": {"type": "array", "items": _obj({"id": {"type": "string"}, "p": {"type": "number"}})},
    "points": {"type": "array", "items": _obj({"claim": {"type": "string"}, "cites": CITES})}})


def _evidence_block(evidence: list[str], prior: EventPrior | None) -> str:
    lines = [f"{i}. {t}" for i, t in enumerate(evidence, start=1)]
    table = "0. Event table: " + (
        f"{prior.event_type}: question '{prior.question}', prior "
        + (f"{prior.prior:.0%} from {len(prior.cases)} cases" if prior.usable else "not yet curated (no number)")
        if prior else "no entry")
    return table + "\n" + "\n".join(lines)


def _valid(cites, n_evidence: int) -> bool:
    return bool(cites) and all(0 <= c <= n_evidence for c in cites)


# --------------------------------------------------------------------------- #
# Aggregation (pure functions; tested without the API)
# --------------------------------------------------------------------------- #

def normalize(probs: dict[str, float], ids: list[str]) -> dict[str, float]:
    v = np.array([max(1e-4, min(1 - 1e-4, float(probs.get(i, 0.0)))) for i in ids])
    v = v / v.sum()
    return dict(zip(ids, v.tolist()))


def aggregate(persona_probs: dict[str, dict[str, float]], ids: list[str]) -> tuple[dict[str, float], float]:
    """Geometric mean of odds per scenario, renormalised; returns (probs, mean persona std-dev)."""
    mat = np.array([[persona_probs[p][i] for i in ids] for p in persona_probs])
    odds = mat / (1 - mat)
    gm = np.exp(np.log(odds).mean(axis=0))
    probs = gm / (1 + gm)
    probs = probs / probs.sum()
    spread = float(mat.std(axis=0).mean()) if len(mat) > 1 else 0.0
    return dict(zip(ids, probs.tolist())), spread


def disagreement_label(spread: float) -> str:
    return "high" if spread >= 0.12 else "medium" if spread >= 0.06 else "low"


def shrink_to_prior(agg: dict[str, float], yes_ids: list[str], prior: EventPrior | None,
                    k: float = SHRINK) -> tuple[dict[str, float], bool]:
    """Pull the 'yes' group's total toward a curated prior, keeping shares within groups."""
    if prior is None or not prior.usable or not yes_ids or len(yes_ids) == len(agg):
        return agg, False
    p_yes = sum(agg[i] for i in yes_ids)
    target = (1 - k) * p_yes + k * prior.prior
    out = {}
    for i, p in agg.items():
        out[i] = p / p_yes * target if i in yes_ids else p / (1 - p_yes) * (1 - target)
    return out, True


def expected_price(agg: dict[str, float], scenarios: list[dict]) -> float | None:
    prices = {s["id"]: s.get("price") for s in scenarios}
    if any(prices.get(i) in (None, 0) for i in agg):
        return None
    return float(sum(p * prices[i] for i, p in agg.items()))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_panel(ticker: str, company: str, event_type: str, evidence: list[str],
              prior: EventPrior | None, reference_price: float | None, today: str,
              trigger: str = "new", client=None, provider: str = "anthropic") -> PanelRun:
    if client is None and provider == "deepseek":
        client = "deepseek"
        model = LLM.deepseek_model("panel")
    else:
        import anthropic

        client = client or anthropic.Anthropic()
        model = PANEL_MODEL
    evidence = evidence[:30]
    n = len(evidence)
    ev = _evidence_block(evidence, prior)
    ctx = (f"Company: {company} ({ticker}). Event type: {event_type}. "
           f"Reference price (pre-market or last close): {reference_price if reference_price else 'unknown'}.")
    dropped = 0

    a = _call(client, f"You map the actors in a major corporate event.\n{COMMON_RULES}",
              f"{ctx}\n\nEvidence:\n{ev}\n\nList the key actors: what each wants, what they can do "
              "(votes, vetoes, authority), how credible their statements are versus past actions, and what limits them.",
              ACTORS_SCHEMA)
    actors = [x for x in a["actors"] if _valid(x["cites"], n)]
    dropped += len(a["actors"]) - len(actors)

    s = _call(client, f"You build scenario trees for corporate events.\n{COMMON_RULES}",
              f"{ctx}\n\nEvidence:\n{ev}\n\nActors:\n{json.dumps(actors)}\n\n"
              "List 3 to 5 mutually exclusive outcomes, each with a time frame. Break multi-step paths into "
              "stages (announce, approve, close). Estimate the share price under each outcome. Mark "
              "yes_for_prior=true for outcomes that answer 'yes' to the event table's question. Then list 3 to 5 "
              "tripwires: future signals that would move these probabilities, each with 1 to 3 short keyword "
              "phrases that would appear in a headline about it. Use short ids like s1, s2.",
              SCENARIO_SCHEMA)
    scenarios = [x for x in s["scenarios"] if _valid(x["cites"], n)][:5]
    dropped += len(s["scenarios"]) - len(scenarios)
    if len(scenarios) < 2:
        raise RuntimeError("panel produced fewer than 2 cited scenarios")
    ids = [x["id"] for x in scenarios]
    scen_txt = "\n".join(f"{x['id']}: {x['label']} ({x['timeframe']})" for x in scenarios)

    personas: dict[str, dict] = {}
    for name, role in PERSONAS.items():
        sees = (f"Evidence:\n{_evidence_block([], prior).rstrip()}" if name == "statistician"
                else f"Evidence:\n{ev}")
        out = _call(client, f"You are {role}.\n{COMMON_RULES}",
                    f"{ctx}\n\n{sees}\n\nScenarios (mutually exclusive):\n{scen_txt}\n\n"
                    "Give your probability for each scenario id (they should sum to 1) and the key points behind it.",
                    PERSONA_SCHEMA)
        limit = 0 if name == "statistician" else n
        points = [p for p in out["points"] if _valid(p["cites"], limit)]
        dropped += len(out["points"]) - len(points)
        personas[name] = {"probs": normalize({x["id"]: x["p"] for x in out["probabilities"]}, ids),
                          "points": points}

    agg, spread = aggregate({k: v["probs"] for k, v in personas.items()}, ids)
    agg, shrunk = shrink_to_prior(agg, [x["id"] for x in scenarios if x.get("yes_for_prior")], prior)
    return PanelRun(
        date=today, trigger=trigger, model=model, prompt_version=PANEL_PROMPT_VERSION,
        evidence=evidence, actors=actors, scenarios=scenarios, personas=personas, aggregate=agg,
        disagreement=spread, disagreement_label=disagreement_label(spread), shrunk_to_prior=shrunk,
        expected_price=expected_price(agg, scenarios), reference_price=reference_price,
        tripwires=s["tripwires"][:5], dropped_claims=dropped)


@dataclass
class PanelOutcome:
    record: EventRecord
    ran_today: bool
    note: str


def process_events(ticker: str, company: str, news, priors: dict[str, EventPrior],
                   reference_price: float | None, data_dir: str | Path, today: str,
                   llm_ok, client=None) -> tuple[list[PanelOutcome], list[str]]:
    """llm_ok: provider name ("deepseek" / "anthropic"), True (= anthropic) or False/None (skip)."""
    provider = "anthropic" if llm_ok is True else (llm_ok or None)
    """Run, re-run (tripwire) or show the panel for each live materiality-3 event."""
    warnings: list[str] = []
    outcomes: list[PanelOutcome] = []
    records = load_events(data_dir, ticker)
    heads = [{"title": h.title, "published_utc": h.published_utc} for h in news.headlines] if news else []
    major = {}
    if news:
        for h in news.headlines:
            if h.materiality >= 3 and h.event_type != "none":
                major.setdefault(h.event_type, []).append(h.title)
    types = set(major) | {r.event_type for r in records
                          if (date.fromisoformat(today) - date.fromisoformat(r.first_seen)).days <= ACTIVE_DAYS}
    for et in sorted(types):
        rec = active_event(records, et, today)
        trigger = None
        if rec is None:
            if et not in major:
                continue
            rec = EventRecord(f"{ticker}_{et}_{today}", ticker, et, today)
            trigger = "new"
        elif rec.latest is None:
            trigger = "new"
        elif rec.latest.date != today:
            hits = fired_tripwires(rec, heads)
            if hits:
                trigger = "tripwire: " + "; ".join(hits)
        if trigger is None:
            outcomes.append(PanelOutcome(rec, False, f"last run {rec.latest.date}; no tripwire fired"
                                         if rec.latest else "not run"))
            continue
        if not (provider or client):
            warnings.append(f"materiality-3 event '{et}': analyst panel needs an LLM key (DEEPSEEK_API_KEY)")
            if rec.latest:
                outcomes.append(PanelOutcome(rec, False, f"last run {rec.latest.date}; re-run due ({trigger}) but no API key"))
            continue
        evidence = [h["title"] for h in sorted(heads, key=lambda h: h["published_utc"], reverse=True)]
        evidence = major.get(et, []) + [t for t in evidence if t not in major.get(et, [])]
        if rec.latest:   # a re-run keeps the earlier evidence after today's headlines
            evidence += [t for t in rec.latest.evidence if t not in evidence]
        try:
            run = run_panel(ticker, company, et, evidence, priors.get(et), reference_price, today, trigger,
                            client, provider or "anthropic")
            rec.history.append(run)
            save_event(rec, data_dir)
            outcomes.append(PanelOutcome(rec, True, trigger))
        except Exception as exc:
            log.exception("%s: analyst panel failed for %s", ticker, et)
            warnings.append(f"analyst panel failed for {et}: {type(exc).__name__}: {exc}")
            if rec.latest:
                outcomes.append(PanelOutcome(rec, False, f"re-run failed; showing {rec.latest.date}"))
    return outcomes, warnings
