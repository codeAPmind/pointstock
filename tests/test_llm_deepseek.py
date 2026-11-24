"""DeepSeek provider: selection, request shape, failure handling, news and panel paths (HTTP mocked)."""
import json

import pytest

from premarket import ROOT
from premarket import llm as LLM
from premarket import news as N
from premarket import panel as P
from premarket.baserates import load_event_priors

PRIORS = load_event_priors(ROOT / "event_base_rates.json")
ENV = {"DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1/",
       "DEEPSEEK_MODEL": "deepseek-flash"}


def test_provider_selection():
    assert LLM.active_provider("auto", {"DEEPSEEK_API_KEY": "k"}) == "deepseek"
    assert LLM.active_provider("deepseek", {}) is None
    assert LLM.active_provider("deepseek", {"DEEPSEEK_API_KEY": "k"}) == "deepseek"
    assert LLM.active_provider("anthropic", {"ANTHROPIC_API_KEY": "k"}) == "anthropic"
    assert LLM.deepseek_model("panel", {"DEEPSEEK_MODEL": "a"}) == "a"
    assert LLM.deepseek_model("panel", {"DEEPSEEK_MODEL": "a", "DEEPSEEK_PANEL_MODEL": "b"}) == "b"
    assert LLM.deepseek_model("news", {}) == LLM.DEEPSEEK_DEFAULT_MODEL


def reply(obj, finish="stop", model="deepseek-flash"):
    return {"model": model, "choices": [{"finish_reason": finish,
                                         "message": {"role": "assistant", "content": json.dumps(obj)}}]}


def test_request_shape(monkeypatch):
    seen = {}
    def fake_post(url, payload, key, timeout):
        seen.update(url=url, payload=payload, key=key)
        return reply({"ok": 1})
    monkeypatch.setattr(LLM, "_post", fake_post)
    data, model = LLM.deepseek_json("sys", "user", {"type": "object"}, env=ENV)
    assert data == {"ok": 1} and model == "deepseek-flash"
    assert seen["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert seen["key"] == "test-key"
    p = seen["payload"]
    assert p["model"] == "deepseek-flash" and p["response_format"] == {"type": "json_object"}
    assert "JSON schema" in p["messages"][0]["content"]


@pytest.mark.parametrize("resp,err", [
    (reply({"a": 1}, finish="length"), "length"),
    ({"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}, "empty"),
])
def test_truncated_or_empty_replies_raise(monkeypatch, resp, err):
    monkeypatch.setattr(LLM, "_post", lambda *a: resp)
    with pytest.raises(RuntimeError, match=err):
        LLM.deepseek_json("s", "u", {}, env=ENV)


def test_missing_key_raises():
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        LLM.deepseek_json("s", "u", {}, env={})


def h(title):
    return {"title": title, "source": "Wire", "published_utc": "2026-11-24T05:00:00+00:00", "url": None,
            "summary": "", "feed": "google"}


def test_news_scoring_via_deepseek(monkeypatch, tmp_path):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(LLM, "_post", lambda *a: reply({"items": [
        {"i": 1, "sentiment": 0.9, "materiality": 3, "event_type": "ceo_return", "why": "CEO back"},
        {"i": 2, "sentiment": -2, "materiality": 1, "event_type": "none", "why": "minor"}]}))
    r = N.analyze("NVDA", "Nvidia", ["Nvidia"], [h("Nvidia reinstates Huang as CEO"), h("Nvidia dips")],
                  [], PRIORS, tmp_path, "2026-11-24", provider="deepseek")
    assert r.scorer == "deepseek:deepseek-flash"
    assert r.headlines[1].sentiment == -1.0                       # clipped
    assert "ceo_return" in r.events and not r.warnings


def test_news_deepseek_failure_falls_back_to_lexicon(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    def boom(*a):
        raise RuntimeError("DeepSeek HTTP 401: invalid key")
    monkeypatch.setattr(LLM, "_post", boom)
    r = N.analyze("NVDA", "Nvidia", ["Nvidia"], [h("Nvidia shares soar")], [], PRIORS, tmp_path,
                  "2026-11-24", provider="deepseek")
    assert r.scorer == "lexicon" and any("DeepSeek headline scoring failed" in w for w in r.warnings)


def test_no_provider_uses_lexicon(tmp_path):
    r = N.analyze("NVDA", "Nvidia", ["Nvidia"], [h("Nvidia shares soar")], [], PRIORS, tmp_path,
                  "2026-11-24", provider=None)
    assert r.scorer == "lexicon"


def test_panel_via_deepseek(monkeypatch, tmp_path):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    calls = []

    def fake_post(url, payload, key, timeout):
        calls.append(payload)
        sys_prompt = payload["messages"][0]["content"]
        if '"actors"' in sys_prompt:
            return reply({"actors": [{"name": "Board", "wants": "x", "can_do": "y", "credibility": "z",
                                      "constraints": "w", "cites": [1]}]})
        if '"scenarios"' in sys_prompt:
            return reply({"scenarios": [
                {"id": "s1", "label": "Holds", "timeframe": "20d", "stages": [], "price": 170, "yes_for_prior": True, "cites": [1]},
                {"id": "s2", "label": "Fades", "timeframe": "20d", "stages": [], "price": 150, "yes_for_prior": False, "cites": [1]}],
                "tripwires": [{"signal": "Board statement", "keywords": ["board statement"]}]})
        return reply({"probabilities": [{"id": "s1", "p": 0.6}, {"id": "s2", "p": 0.4}],
                      "points": [{"claim": "c", "cites": [0]}]})

    monkeypatch.setattr(LLM, "_post", fake_post)
    from premarket.news import NewsResult, ScoredHeadline
    nr = NewsResult("NVDA", "deepseek:deepseek-flash", "v",
                    [ScoredHeadline("Board reinstates Huang", "W", "2026-11-24T05:00:00+00:00", None, 0.8, 3, "ceo_return", "")],
                    1, 1, 0.8, None, {"ceo_return": ["Board reinstates Huang"]})
    out, w = P.process_events("NVDA", "Nvidia", nr, PRIORS, 158.0, tmp_path, "2026-11-24", "deepseek")
    assert not w and out[0].ran_today
    run = out[0].record.latest
    assert run.model == "deepseek-flash" and len(calls) == 2 + len(P.PERSONAS)
    assert run.aggregate["s1"] == pytest.approx(0.6)
