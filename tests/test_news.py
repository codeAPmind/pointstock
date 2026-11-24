"""News tier 1: relevance, de-duplication, scorers, LLM output parsing, offline cache."""
import json
from types import SimpleNamespace

import pytest

from premarket import ROOT
from premarket import data as D
from premarket import news as N
from premarket.baserates import load_event_priors

PRIORS = load_event_priors(ROOT / "event_base_rates.json")


def h(title, ts="2026-11-24T05:00:00+00:00", source="Wire"):
    return {"title": title, "source": source, "published_utc": ts, "url": None, "summary": "", "feed": "google"}


def test_relevance_filter():
    assert N.is_relevant("Opendoor Technologies Drops 14%", "OPEN", ["Opendoor"])
    assert not N.is_relevant("Judges Open Door for Trump to Win Appeal", "OPEN", ["Opendoor"])
    assert N.is_relevant("Short sellers pile into $OPEN", "OPEN", ["Opendoor"])
    assert N.is_relevant("NVDA slides premarket", "NVDA", ["Nvidia"])
    assert not N.is_relevant("10 AI stocks to watch, including Nvidia", "NVDA", ["Nvidia"])
    assert not N.is_relevant("Stocks making the biggest moves premarket: Nvidia, Tesla", "NVDA", ["Nvidia"])


def test_dedupe_on_first_80_chars_lowercase():
    a = "Nvidia shares jump after board reinstates Jensen Huang as chief executive officer, sources say"
    b = a.upper()[:85] + " - extra tail that differs"
    out = N.dedupe([h(b, "2026-11-24T06:00:00+00:00"), h(a, "2026-11-24T05:00:00+00:00")])
    assert len(out) == 1 and out[0]["title"] == a   # earliest copy kept


def test_lexicon_scorer_signs_and_events():
    s = N.score_lexicon([h("Jensen Huang's return to Nvidia sends shares soaring"),
                         h("Tesla recalls 200,000 vehicles, shares slide"),
                         h("Board reinstates Huang as Nvidia CEO")], PRIORS)
    assert s[0].sentiment > 0 and s[1].sentiment < 0
    assert s[2].event_type == "ceo_return" and s[2].materiality == 3


def test_daily_score_is_materiality_weighted():
    sc = [N.ScoredHeadline("a", "s", "t", None, 1.0, 3, "none", ""),
          N.ScoredHeadline("b", "s", "t", None, -1.0, 1, "none", ""),
          N.ScoredHeadline("c", "s", "t", None, 0.9, 0, "none", "")]
    assert N.daily_score(sc) == pytest.approx((3 - 1) / 4)
    assert N.daily_score(sc[2:]) is None


def test_news_probability_mapping():
    assert N.news_probability(None, 0.52) is None
    assert N.news_probability(0.0, 0.52) == pytest.approx(0.52)
    assert 0.52 < N.news_probability(1.0, 0.52) < 0.63


def test_parse_llm_items_clips_and_validates():
    items = [h("A"), h("B"), h("C")]
    text = json.dumps({"items": [
        {"i": 1, "sentiment": 1.7, "materiality": 5, "event_type": "ceo_return", "why": "x"},
        {"i": 3, "sentiment": -0.4, "materiality": 1, "event_type": "made_up", "why": "y"},
        {"i": 9, "sentiment": 0.1, "materiality": 1, "event_type": "none", "why": "ignored"}]})
    out = N.parse_llm_items(text, items, list(PRIORS))
    assert [o.title for o in out] == ["A", "C"]
    assert out[0].sentiment == 1.0 and out[0].materiality == 3
    assert out[1].event_type == "none"


class FakeClient:
    """Stands in for anthropic.Anthropic(); records the request."""
    def __init__(self, payload, stop="end_turn"):
        self.payload, self.stop, self.calls = payload, stop, []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(stop_reason=self.stop,
                               content=[SimpleNamespace(type="text", text=json.dumps(self.payload))])


def test_analyze_with_llm_client(tmp_path):
    raw = [h("Nvidia reinstates Huang as CEO"), h("Nvidia: 5 things to know", source="X"),
           h("Weather is nice today")]
    fake = FakeClient({"items": [{"i": 1, "sentiment": 0.8, "materiality": 3, "event_type": "ceo_return", "why": "CEO back"},
                                 {"i": 2, "sentiment": 0.1, "materiality": 1, "event_type": "none", "why": "minor"}]})
    r = N.analyze("NVDA", "Nvidia", ["Nvidia"], raw, [], PRIORS, tmp_path, "2026-11-24", client=fake)
    assert r.scorer == N.LLM_MODEL and r.n_relevant == 2
    assert "ceo_return" in r.events
    call = fake.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert (tmp_path / "news" / "2026-11-24_NVDA.json").exists()


def test_llm_refusal_falls_back_to_lexicon(tmp_path):
    fake = FakeClient({}, stop="refusal")
    r = N.analyze("NVDA", "Nvidia", ["Nvidia"], [h("Nvidia shares soar")], [], PRIORS, tmp_path,
                  "2026-11-24", client=fake)
    assert r.scorer == "lexicon" and r.headlines[0].sentiment > 0
    assert any("lexicon" in w for w in r.warnings)


def test_offline_news_uses_cache_and_warns(tmp_path):
    from datetime import datetime, timezone
    now = datetime(2026, 11, 24, 13, 0, tzinfo=timezone.utc)
    (tmp_path / "news").mkdir()
    cached = [h("Nvidia shares soar", "2026-11-24T05:00:00+00:00") | {"feed": "yahoo"},
              h("Old Nvidia story", "2026-11-20T05:00:00+00:00") | {"feed": "google"}]
    (tmp_path / "news" / "2026-11-23_NVDA_raw.json").write_text(json.dumps(cached))
    items, failed = D.get_news("NVDA", "Nvidia", tmp_path, offline=True, now_utc=now)
    assert set(failed) == {"yahoo", "google"}
    assert [i["title"] for i in items] == ["Nvidia shares soar"]    # 36-hour window applied
    r = N.analyze("NVDA", "Nvidia", ["Nvidia"], items, failed, PRIORS, tmp_path, "2026-11-24", use_llm=False)
    assert any("news source failed" in w for w in r.warnings)
