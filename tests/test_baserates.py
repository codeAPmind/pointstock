"""Base rates: correct counting, no look-ahead, gap matching, event priors."""
import numpy as np
import pandas as pd
import pytest

from premarket import ROOT
from premarket import baserates as B
from premarket.features import build_features
from tests.conftest import synthetic_ohlcv


@pytest.fixture
def px():
    return synthetic_ohlcv(n=1500, seed=7)


def test_conditional_rates_only_report_active_conditions(px):
    feats = build_features(px)
    active = B.condition_frame(px, feats).iloc[-1]
    got = {r.key for r in B.conditional_rates(px, feats)}
    assert got == set(active.index[active])


def test_conditional_rates_match_brute_force(px):
    feats = build_features(px)
    conds = B.condition_frame(px, feats)
    c = px["Close"]
    for r in B.conditional_rates(px, feats):
        days = [d for d in conds.index[:-1] if conds.at[d, r.key]]
        outcomes = []
        for d in days:
            i = c.index.get_loc(d)
            outcomes.append(c.iloc[i + 1] / c.iloc[i] - 1)
        assert r.n == len(outcomes)
        assert r.p_up_1d == pytest.approx(np.mean(np.array(outcomes) > 0))
        assert r.mean_ret_1d == pytest.approx(np.mean(outcomes))
        assert r.usable == (r.n >= B.MIN_CASES)


def test_conditional_rates_ignore_the_future(px):
    """Rates as of day t are identical whether or not later days exist."""
    feats = build_features(px)
    asof = px.index[1000]
    a = B.conditional_rates(px, feats, asof=asof)
    b = B.conditional_rates(px.loc[:asof], build_features(px.loc[:asof]))
    assert [vars(x) for x in a] == [vars(x) for x in b]


def test_five_day_outcomes_exclude_unfinished_windows(px):
    """A case 3 days before `asof` has a 1-day outcome but no 5-day outcome yet."""
    feats = build_features(px)
    for r in B.conditional_rates(px, feats):
        assert r.n_5d <= r.n


def test_gap_rates_threshold_and_matching(px):
    assert B.gap_rates(px, 0.005) is None
    assert B.gap_rates(px, None) is None
    g = B.gap_rates(px, 0.02, sigma_today=0.02)
    pg = B.past_gaps(px)
    m = pg[(pg.gap >= 0.012) & (pg.gap <= 0.032)]
    assert g.n == len(m) and (g.lo, g.hi) == pytest.approx((0.012, 0.032))
    assert g.p_close_above_open == pytest.approx((m.oc > 0).mean())
    assert g.p_gap_fills == pytest.approx(m.fills.mean())


def test_negative_gap_matches_same_sign_only(px):
    g = B.gap_rates(px, -0.02)
    assert g.lo == pytest.approx(-0.032) and g.hi == pytest.approx(-0.012)


def test_gap_fill_definition():
    """Up gap fills if the low touches the prior close; down gap if the high does."""
    n = 30
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"Open": c, "High": c * 1.005, "Low": c * 0.995, "Close": c, "Volume": 1.0},
                      index=pd.bdate_range("2024-01-01", periods=n))
    pc = df["Close"].iloc[-3]
    # Day -2: gap up 3%, low stays above prior close -> no fill.
    df.iloc[-2] = [pc * 1.03, pc * 1.04, pc * 1.01, pc * 1.02, 1.0]
    pc2 = df["Close"].iloc[-2]
    # Day -1: gap down 3%, high reaches the prior close -> fill.
    df.iloc[-1] = [pc2 * 0.97, pc2 * 1.001, pc2 * 0.96, pc2 * 0.99, 1.0]
    pg = B.past_gaps(df)
    assert pg["gap"].iloc[-2] == pytest.approx(0.03)
    assert not pg["fills"].iloc[-2]
    assert pg["fills"].iloc[-1]
    assert pg["oc"].iloc[-1] == pytest.approx(0.99 / 0.97 - 1)


def test_widened_band_contains_garch_band(px):
    g = B.gap_rates(px, 0.03, sigma_today=0.02)
    assert g.usable and g.band68 and g.band95
    w = B.widened_bands(100.0, (98, 102), (96, 104), g)
    (a68, b68), (a95, b95) = w
    assert a68 <= 98 and b68 >= 102 and a95 <= 96 and b95 >= 104
    assert a95 <= a68 and b95 >= b68
    assert B.widened_bands(100.0, (98, 102), (96, 104), None) is None


def test_event_priors_placeholders_are_never_usable():
    priors = B.load_event_priors(ROOT / "event_base_rates.json")
    assert {"ceo_departure", "ceo_return", "merger_talks", "regulatory_legal",
            "earnings", "analyst_action"} <= set(priors)
    assert not any(p.usable for p in priors.values())


def test_detect_events_keyword_match():
    priors = B.load_event_priors(ROOT / "event_base_rates.json")
    hits = B.detect_events(["Board reinstates Huang as CEO", "Analyst upgrades NVDA",
                            "10 stocks to watch this week"], priors)
    assert "ceo_return" in hits and "analyst_action" in hits
    assert "merger_talks" not in hits
