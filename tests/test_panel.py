"""Analyst panel: aggregation math, prior shrink, citation rule, tripwires, event history."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from premarket import ROOT
from premarket import panel as P
from premarket.baserates import EventPrior, load_event_priors
from premarket.news import NewsResult, ScoredHeadline

PRIORS = load_event_priors(ROOT / "event_base_rates.json")


def test_aggregate_is_geometric_mean_of_odds():
    ids = ["a", "b"]
    probs, spread = P.aggregate({"x": {"a": 0.8, "b": 0.2}, "y": {"a": 0.5, "b": 0.5}}, ids)
    odds_a = np.sqrt(4.0 * 1.0)          # geometric mean of 4 and 1 = 2 -> p = 2/3
    odds_b = np.sqrt(0.25 * 1.0)         # 0.5 -> p = 1/3
    pa, pb = odds_a / (1 + odds_a), odds_b / (1 + odds_b)
    assert probs["a"] == pytest.approx(pa / (pa + pb))
    assert sum(probs.values()) == pytest.approx(1.0)
    assert spread == pytest.approx(0.15)
    assert P.disagreement_label(spread) == "high"


def test_identical_personas_have_no_disagreement():
    probs, spread = P.aggregate({k: {"a": 0.7, "b": 0.3} for k in "pqr"}, ["a", "b"])
    assert probs["a"] == pytest.approx(0.7) and spread == pytest.approx(0.0, abs=1e-12)
    assert P.disagreement_label(0.0) == "low"


def test_normalize_handles_missing_and_bad_values():
    out = P.normalize({"a": 2.0, "b": -1}, ["a", "b", "c"])
    assert sum(out.values()) == pytest.approx(1.0) and all(v > 0 for v in out.values())


def test_shrink_only_with_curated_prior():
    agg = {"s1": 0.8, "s2": 0.2}
    placeholder = PRIORS["ceo_return"]
    assert P.shrink_to_prior(agg, ["s1"], placeholder) == (agg, False)
    curated = EventPrior("ceo_return", "q", [], 0.4, True, cases=[{"x": 1}])
    out, shrunk = P.shrink_to_prior(agg, ["s1"], curated)
    assert shrunk and out["s1"] == pytest.approx(0.6) and out["s2"] == pytest.approx(0.4)


def test_expected_price():
    sc = [{"id": "a", "price": 100.0}, {"id": "b", "price": 80.0}]
    assert P.expected_price({"a": 0.75, "b": 0.25}, sc) == pytest.approx(95.0)
    assert P.expected_price({"a": 1.0}, [{"id": "a", "price": None}]) is None


class FakePanelClient:
    """Answers each panel step by the schema it is asked for; records calls."""
    def __init__(self, persona_probs=None):
        self.calls = []
        self.persona_probs = persona_probs or {"s1": 0.6, "s2": 0.3, "s3": 0.1}
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        props = kw["output_config"]["format"]["schema"]["properties"]
        if "actors" in props:
            out = {"actors": [
                {"name": "Board", "wants": "stability", "can_do": "appoint CEO", "credibility": "high",
                 "constraints": "fiduciary duty", "cites": [1]},
                {"name": "Rumour", "wants": "?", "can_do": "?", "credibility": "?", "constraints": "?", "cites": []}]}
        elif "scenarios" in props:
            out = {"scenarios": [
                {"id": "s1", "label": "Rally holds", "timeframe": "20d", "stages": [], "price": 180.0,
                 "yes_for_prior": True, "cites": [1, 2]},
                {"id": "s2", "label": "Partial giveback", "timeframe": "20d", "stages": [], "price": 160.0,
                 "yes_for_prior": False, "cites": [2]},
                {"id": "s3", "label": "Full giveback", "timeframe": "20d", "stages": [], "price": 140.0,
                 "yes_for_prior": False, "cites": [3]}],
                "tripwires": [{"signal": "SEC probe update", "keywords": ["sec probe"]},
                              {"signal": "Board statement", "keywords": ["board statement"]}]}
        else:
            out = {"probabilities": [{"id": k, "p": v} for k, v in self.persona_probs.items()],
                   "points": [{"claim": "supported", "cites": [1]}, {"claim": "made up", "cites": [99]}]}
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text=json.dumps(out))])


def news(titles, mat=3, et="ceo_return", ts="2026-11-24T05:00:00+00:00"):
    hs = [ScoredHeadline(t, "Wire", ts, None, 0.5, mat, et, "") for t in titles]
    return NewsResult("NVDA", "lexicon", "v", hs, len(hs), len(hs), 0.5, None, {et: titles})


def test_run_panel_end_to_end_with_fake_client():
    fake = FakePanelClient()
    run = P.run_panel("NVDA", "Nvidia", "ceo_return", ["Board reinstates Huang", "Shares soar", "Critics wary"],
                      PRIORS["ceo_return"], 158.0, "2026-11-24", client=fake)
    assert len(fake.calls) == 2 + len(P.PERSONAS)
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"]["effort"] == "high"
    assert [a["name"] for a in run.actors] == ["Board"]           # uncited actor dropped
    assert run.dropped_claims >= 1 + 4                             # rumour + bad cites
    assert run.aggregate["s1"] == pytest.approx(0.6, abs=1e-6)     # identical personas
    assert run.disagreement_label == "low" and not run.shrunk_to_prior
    assert run.expected_price == pytest.approx(0.6 * 180 + 0.3 * 160 + 0.1 * 140, rel=1e-4)
    # The statistician never sees the headlines.
    stat = fake.calls[2 + list(P.PERSONAS).index("statistician")]
    assert "Board reinstates Huang" not in stat["messages"][0]["content"]


def test_process_events_new_then_cached_then_tripwire(tmp_path):
    fake = FakePanelClient()
    n1 = news(["Board reinstates Huang as Nvidia CEO", "Shares soar"])
    out, w = P.process_events("NVDA", "Nvidia", n1, PRIORS, 158.0, tmp_path, "2026-11-24", True, fake)
    assert len(out) == 1 and out[0].ran_today and out[0].record.latest.trigger == "new"
    assert (tmp_path / "events" / "NVDA_ceo_return_2026-11-24.json").exists()

    # Next day, no tripwire in the news: show the saved panel, no API calls.
    calls = len(fake.calls)
    n2 = news(["Nvidia keeps gains"], mat=1, et="none", ts="2026-11-25T05:00:00+00:00")
    out, _ = P.process_events("NVDA", "Nvidia", n2, PRIORS, 160.0, tmp_path, "2026-11-25", True, fake)
    assert not out[0].ran_today and len(fake.calls) == calls

    # A tripwire headline re-runs the panel and appends to the history.
    n3 = news(["SEC probe widens into Nvidia board"], mat=2, et="regulatory_legal",
              ts="2026-11-26T05:00:00+00:00")
    out, _ = P.process_events("NVDA", "Nvidia", n3, PRIORS, 150.0, tmp_path, "2026-11-26", True, fake)
    rec = [o for o in out if o.record.event_type == "ceo_return"][0].record
    assert len(rec.history) == 2 and rec.latest.trigger.startswith("tripwire: SEC probe")


def test_no_api_key_skips_panel_with_warning(tmp_path):
    out, w = P.process_events("NVDA", "Nvidia", news(["Board reinstates Huang"]), PRIORS, 158.0,
                              tmp_path, "2026-11-24", llm_ok=False)
    assert out == [] and any("needs an LLM key" in x for x in w)


def test_low_materiality_does_not_trigger(tmp_path):
    out, w = P.process_events("NVDA", "Nvidia", news(["Analyst nudges target"], mat=2, et="analyst_action"),
                              PRIORS, 158.0, tmp_path, "2026-11-24", True, FakePanelClient())
    assert out == [] and w == []
