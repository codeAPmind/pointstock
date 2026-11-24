"""Forecast ledger: upsert, outcome filling (yesterday's row gets its outcome), scoring."""
import numpy as np
import pandas as pd
import pytest

from premarket import ledger as L


def row(date, ticker="NVDA", last_date=None, last_close=100.0, p_final=0.55, p_uncond=0.52, **kw):
    r = {c: None for c in L.COLUMNS}
    r.update(date=date, ticker=ticker, last_close_date=last_date or str(pd.Timestamp(date) - pd.offsets.BDay(1))[:10],
             last_close=last_close, p_uncond=p_uncond, p_prior=p_uncond, p_final=p_final,
             band68_low=last_close * 0.98, band68_high=last_close * 1.02)
    r.update(kw)
    return r


def prices(closes: dict[str, float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in closes])
    return pd.DataFrame({"Close": list(closes.values())}, index=idx)


def test_yesterdays_row_gets_outcome_filled():
    led = L.upsert(L.load("/nonexistent"), [row("2026-11-23", last_date="2026-11-20")])
    px = {"NVDA": prices({"2026-11-20": 100.0, "2026-11-23": 101.5})}
    led, n = L.fill_outcomes(led, px)
    assert n == 1
    r = led.iloc[0]
    assert r.outcome_up == 1.0 and r.actual_close == 101.5
    assert r.actual_ret == pytest.approx(np.log(1.015))
    assert r.inside_band68 == 1.0


def test_todays_row_waits_for_the_close():
    led = L.upsert(L.load("/nonexistent"), [row("2026-11-24", last_date="2026-11-23")])
    led, n = L.fill_outcomes(led, {"NVDA": prices({"2026-11-23": 100.0})})
    assert n == 0 and pd.isna(led.iloc[0].outcome_up)


def test_outcome_is_split_safe():
    """After a 4:1 split the adjusted series shows 25 -> 26; the row stored 100."""
    led = L.upsert(L.load("/nonexistent"), [row("2026-11-23", last_date="2026-11-20", last_close=100.0)])
    led, _ = L.fill_outcomes(led, {"NVDA": prices({"2026-11-20": 25.0, "2026-11-23": 26.0})})
    r = led.iloc[0]
    assert r.outcome_up == 1.0 and r.actual_ret == pytest.approx(np.log(26 / 25))
    assert r.inside_band68 == 0.0   # +4% is outside the ±2% band, scaled to the split


def test_upsert_replaces_same_day_row(tmp_path):
    led = L.upsert(L.load(tmp_path), [row("2026-11-24", p_final=0.5), row("2026-11-24", "TSLA")])
    led = L.upsert(led, [row("2026-11-24", p_final=0.6)])
    assert len(led) == 2
    assert led[led.ticker == "NVDA"].p_final.item() == 0.6
    L.save(led, tmp_path)
    back = L.load(tmp_path)
    assert len(back) == 2 and list(back.columns) == L.COLUMNS


def test_score_and_disable_bad_signal():
    rng = np.random.default_rng(0)
    rows = []
    for i, d in enumerate(pd.bdate_range("2026-01-01", periods=80)):
        up = float(rng.random() < 0.5)
        rows.append(row(str(d.date()), p_final=0.5, p_uncond=0.5, p_news=0.9 if up == 0 else 0.1,
                        p_ml=0.5, outcome_up=up, inside_band68=1.0))
    sc = L.score(pd.DataFrame(rows, columns=L.COLUMNS))
    assert sc.n_scored == 80
    assert sc.bss["p_final"] == pytest.approx(0.0)
    assert sc.bss["p_news"] < 0 and "p_news" in sc.disabled
    assert "p_ml" not in sc.disabled
    assert sc.band_coverage == 1.0
    assert "p_final BSS" in sc.summary()


def test_empty_ledger_scorecard():
    sc = L.score(L.load("/nonexistent"))
    assert sc.n_scored == 0 and sc.disabled == []
    assert "no scored forecasts" in sc.summary()
