"""Combiner (DESIGN.md tests): all weights 0 -> p_final == p0; output inside the 35-65% cap."""
import math

import pytest

from premarket import combine as C


def sig(p, w, ref=0.52, name="x"):
    return C.Signal(name, name, p, ref, w)


@pytest.mark.parametrize("p0", [0.40, 0.46, 0.52, 0.60])
def test_zero_weights_return_prior(p0):
    out = C.combine(p0, [sig(0.9, 0.0), sig(0.1, 0.0), sig(0.6, 0.0)])
    assert out.p_final == pytest.approx(p0)
    assert out.agreement == (0, 0, 0)


def test_no_signals_return_prior():
    assert C.combine(0.53, []).p_final == pytest.approx(0.53)


def test_output_always_inside_cap():
    hi = C.combine(0.6, [sig(0.99, 1.0), sig(0.99, 1.0)])
    lo = C.combine(0.4, [sig(0.01, 1.0), sig(0.01, 1.0)])
    assert hi.p_final == 0.65 and hi.capped and hi.p_raw > 0.65
    assert lo.p_final == 0.35 and lo.capped


def test_signal_at_its_reference_is_neutral():
    """A signal equal to its own no-information reference moves nothing, even on a gap day."""
    out = C.combine(0.42, [sig(0.52, 0.5, ref=0.52)], p0_source="gap")
    assert out.p_final == pytest.approx(0.42)


def test_log_odds_arithmetic():
    out = C.combine(0.5, [sig(0.6, 0.5, ref=0.5)])
    expected = 1 / (1 + math.exp(-0.5 * math.log(0.6 / 0.4)))
    assert out.p_final == pytest.approx(expected)


def test_agreement_counts_only_weighted_signals():
    out = C.combine(0.5, [sig(0.6, 0.3), sig(0.4, 0.2), sig(0.7, 0.0)])
    assert out.agreement == (1, 1, 2)


def test_weight_rules():
    assert C.ml_weight(0.01, True) == pytest.approx(0.2)
    assert C.ml_weight(0.5, True) == 0.6
    assert C.ml_weight(0.05, False) == 0.0
    assert C.ml_weight(-0.01, True) == 0.0
    assert C.condition_weight(100) == pytest.approx(0.5)
    assert C.condition_weight(25) == pytest.approx(0.25)
    assert C.condition_weight(10000) == 0.5


def test_baserate_only_uses_condition_signals():
    s = [C.Signal("cond:a", "a", 0.6, 0.5, 0.5), C.Signal("news", "n", 0.9, 0.5, 0.35)]
    p = C.baserate_only(0.5, s)
    assert p == pytest.approx(C.sigmoid(0.5 * C.logit(0.6)))
    assert C.baserate_only(0.5, s[1:]) is None


def test_gap_prior_outside_range_is_not_forced_toward_50():
    """A -4% gap with a 14% gap base rate: the cap follows the prior (p0 ± 15 points)."""
    assert C.cap_for(0.52) == (0.35, 0.65)
    assert C.cap_for(0.14) == pytest.approx((0.02, 0.29))
    out = C.combine(0.14, [sig(0.58, 0.35, ref=0.53)], p0_source="gap")
    assert 0.14 < out.p_final < 0.29 and not out.capped
    strong = C.combine(0.14, [sig(0.99, 1.0, ref=0.5), sig(0.99, 1.0, ref=0.5)])
    assert strong.p_final == pytest.approx(0.29) and strong.capped
