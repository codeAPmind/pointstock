"""Macro context: FMP parsing and surprise signs, auctions, gap split, event-day volatility, scoring, wiring."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from premarket import ledger as L
from premarket import llm as LLM
from premarket import macro as MC
from premarket import run as R


def test_fmp_parsing_dates_and_hawkish_sign():
    rows = [
        # 13:30 UTC = 08:30 EST on 2026-12-10
        {"country": "US", "event": "Core Inflation Rate MoM (Nov)", "date": "2026-12-10 13:30:00",
         "estimate": 0.3, "actual": 0.5, "previous": 0.3, "unit": "%", "impact": "High"},
        {"country": "US", "event": "Unemployment Rate (Nov)", "date": "2026-12-10 13:30:00",
         "estimate": 4.2, "actual": 4.4, "previous": 4.2, "unit": "%"},
        {"country": "US", "event": "Non Farm Payrolls (Nov)", "date": "2026-12-10 13:30:00",
         "estimate": 150, "actual": None, "unit": "K"},
        {"country": "US", "event": "Building Permits (Nov)", "date": "2026-12-10 13:30:00", "estimate": 1.4},
        {"country": "EU", "event": "Inflation Rate MoM", "date": "2026-12-10 10:00:00", "estimate": 0.1, "actual": 0.2},
        # 03:00 UTC on the 11th is still the 10th in New York
        {"country": "US", "event": "Fed Interest Rate Decision", "date": "2026-12-11 03:00:00", "estimate": 4.0},
        {"country": "US", "event": "Core PPI MoM", "date": "2026-12-11 13:30:00", "estimate": 0.2, "actual": 0.2},
    ]
    ev = MC.parse_fmp_calendar(rows, "2026-12-10")
    keys = [(e.key, e.time_et) for e in ev]
    assert keys == [("cpi", "08:30"), ("nfp", "08:30"), ("nfp", "08:30"), ("fomc", "22:00")]
    cpi, unrate, nfp = ev[0], ev[1], ev[2]
    assert cpi.surprise_z == pytest.approx(2.0)          # hotter inflation = hawkish +
    assert unrate.surprise_z == pytest.approx(-2.0)      # higher unemployment = dovish -
    assert nfp.surprise_z is None                        # not released yet
    assert MC.group_surprise(ev) == {"cpi": pytest.approx(2.0), "nfp": pytest.approx(-2.0)}


def test_surprise_is_clipped():
    ev = MC.parse_fmp_calendar([{"country": "US", "event": "Inflation Rate MoM", "date": "2026-12-10 13:30:00",
                                 "estimate": 0.1, "actual": 2.0}], "2026-12-10")
    assert ev[0].surprise_z == 4.0


@pytest.mark.parametrize("term,key", [("10-Year", "auction_10y"), ("19-Year 11-Month", "auction_20y"),
                                      ("29-Year 10-Month", "auction_30y"), ("2-Year", "auction_2y"),
                                      ("26-Week", None), ("4-Year", None)])
def test_auction_key(term, key):
    assert MC.auction_key(term) == key


def test_calendar_without_fmp_key(tmp_path, monkeypatch):
    monkeypatch.setattr(MC, "treasury_auctions", lambda day: [MC.MacroEvent("auction_10y", "10-Year Note auction", "13:00", "Treasury")])
    ev, w = MC.todays_calendar("2026-12-09", {"fomc_dates": ["2026-12-09"]}, tmp_path, env={})
    assert {e.key for e in ev} == {"fomc", "auction_10y"}
    assert any("FMP_API_KEY" in x for x in w)
    cached, _ = MC.todays_calendar("2026-12-09", {}, tmp_path, offline=True)
    assert {e.key for e in cached} == {"fomc", "auction_10y"}


def _series(vals, start="2024-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


def test_gap_decomposition_recovers_betas():
    rng = np.random.default_rng(3)
    n = 400
    rm = rng.normal(0, 0.01, n)
    rx = 1.2 * rm + rng.normal(0, 0.008, n)
    ex = rx - 1.2 * rm
    rs = 1.5 * rm + 0.8 * ex + rng.normal(0, 0.002, n)
    px = lambda r: _series(100 * np.exp(np.cumsum(r)))
    g = MC.gap_decomposition(px(rs), px(rm), px(rx), g_stock=-0.03, g_spy=-0.01, g_sector=-0.02, window=300)
    assert g.beta_mkt == pytest.approx(1.5, abs=0.08) and g.beta_sector == pytest.approx(0.8, abs=0.08)
    assert g.market + g.sector + g.stock_specific == pytest.approx(-0.03)
    assert g.market == pytest.approx(g.beta_mkt * -0.01)
    no_sec = MC.gap_decomposition(px(rs), px(rm), None, -0.03, -0.01, None)
    assert no_sec.beta_sector is None and no_sec.sector == 0.0


def test_event_vol_ratio_and_multiplier():
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.01, 1500)
    idx = pd.bdate_range("2020-01-01", periods=1500)
    dates = list(idx[50::40][:30])
    for d in dates:
        r[idx.get_loc(d)] *= 3.0
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    ratio, n = MC.event_vol_ratio(close, dates, idx[-1])
    assert n == 30 and ratio > 2
    # Only events before `asof` count.
    _, n_early = MC.event_vol_ratio(close, dates, dates[10])
    assert n_early == 10
    ev = [MC.MacroEvent("cpi", "CPI", "08:30", "FMP")]
    mult, det = MC.vol_multiplier(close, ev, {"cpi": dates}, idx[-1])
    assert 1.4 < mult <= MC.MAX_VOL_MULT and det[0]["used"]
    mult_few, det_few = MC.vol_multiplier(close, ev, {"cpi": dates[:10]}, idx[-1])
    assert mult_few == 1.0 and not det_few[0]["used"]          # N < 20


def test_widen():
    assert MC.widen(100, (95, 105), 1.0) == pytest.approx((95, 105))
    lo, hi = MC.widen(100, (95, 105), 1.5)
    assert lo < 95 and hi > 105
    assert math.log(lo / 100) == pytest.approx(1.5 * math.log(0.95))


def h(title, ts="2026-11-26T05:00:00+00:00"):
    return {"title": title, "source": "Wire", "published_utc": ts, "url": None, "summary": "", "feed": "google"}


def test_macro_scoring_paths(monkeypatch):
    items = [h("Fed signals pause as inflation cools"), h("Chip export curbs widen"), h("Local bakery opens")]
    none = MC.score_macro_news("NVDA", "Nvidia", "AI chips", items, None)
    assert none.score is None and none.scorer == "unscored" and none.warnings
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setattr(LLM, "_post", lambda *a: {"model": "deepseek-flash", "choices": [{"finish_reason": "stop",
        "message": {"content": json.dumps({"items": [
            {"i": 1, "category": "macro", "sentiment": 0.4, "materiality": 2, "why": "lower yields"},
            {"i": 2, "category": "industry", "sentiment": -1.5, "materiality": 3, "why": "China sales"},
            {"i": 3, "category": "none", "sentiment": 0.0, "materiality": 0, "why": "irrelevant"}]})}}]})
    # newest first: items are re-sorted by time, so give distinct timestamps
    items = [h("Fed signals pause as inflation cools", "2026-11-26T07:00:00+00:00"),
             h("Chip export curbs widen", "2026-11-26T06:00:00+00:00"),
             h("Local bakery opens", "2026-11-26T05:00:00+00:00")]
    mn = MC.score_macro_news("NVDA", "Nvidia", "AI chips", items, "deepseek")
    assert mn.scorer == "deepseek:deepseek-flash" and len(mn.headlines) == 2
    assert mn.headlines[1].sentiment == -1.0
    assert mn.score == pytest.approx((2 * 0.4 + 3 * -1.0) / 5)
    assert MC.macro_probability(0.0, 0.52) == pytest.approx(0.52)
    assert MC.macro_probability(-1.0, 0.52) < 0.52


def test_scoring_failure_gives_weight_zero(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    def boom(*a):
        raise RuntimeError("DeepSeek HTTP 500")
    monkeypatch.setattr(LLM, "_post", boom)
    mn = MC.score_macro_news("NVDA", "Nvidia", "x", [h("Fed holds")], "deepseek")
    assert mn.score is None and "failed" in mn.warnings[0]


def _card(macro_score):
    mn = MC.MacroNews([], 5, "deepseek:x", macro_score)
    m = MC.MacroContext([], [], {}, None, None, None, 1.0, [], {}, mn)
    return SimpleNamespace(prior=(0.52, 1500), gap_rate=None, direction=None, conditions=[], news=None, macro=m)


def test_combiner_includes_macro_signal_and_respects_disable():
    cb = R.combine_card(_card(-0.8), disabled=[])
    sig = [s for s in cb.signals if s.name == "macro"][0]
    assert sig.weight == MC.MACRO_WEIGHT and sig.p < 0.52 and cb.p_final < 0.52
    off = R.combine_card(_card(-0.8), disabled=["p_macro"])
    assert [s for s in off.signals if s.name == "macro"][0].weight == 0.0
    assert off.p_final == pytest.approx(0.52)
    assert not [s for s in R.combine_card(_card(None), []).signals if s.name == "macro"]


def test_ledger_has_macro_columns_and_reads_old_files(tmp_path):
    assert {"p_macro", "w_macro", "macro_events", "macro_vol_mult"} <= set(L.COLUMNS)
    old = [c for c in L.COLUMNS if c not in ("p_macro", "w_macro", "macro_events", "macro_vol_mult")]
    pd.DataFrame([{c: None for c in old} | {"date": "2026-11-24", "ticker": "NVDA"}]).to_csv(
        L.ledger_path(tmp_path), index=False)
    df = L.load(tmp_path)
    assert list(df.columns) == L.COLUMNS and df["p_macro"].isna().all()


def test_length_cutoff_retries_with_half(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    sizes = []

    def fake_post(url, payload, key, timeout):
        n = payload["messages"][1]["content"].count("[Wire]")
        sizes.append(n)
        if n > 10:
            return {"choices": [{"finish_reason": "length", "message": {"content": "{"}}]}
        return {"model": "m", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
            {"items": [{"i": 1, "category": "macro", "sentiment": -0.5, "materiality": 2, "why": "w"}]})}}]}

    monkeypatch.setattr(LLM, "_post", fake_post)
    items = [h(f"Fed headline {i}", f"2026-11-26T{10 + i % 10:02d}:00:00+00:00") for i in range(20)]
    mn = MC.score_macro_news("NVDA", "Nvidia", "x", items, "deepseek")
    assert sizes == [20, 10] and mn.score == pytest.approx(-0.5) and mn.n_fetched == 10
