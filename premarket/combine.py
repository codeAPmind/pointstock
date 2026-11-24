"""Bayesian combiner: one P(up) from the prior plus weighted evidence (DESIGN.md).

    logit(p_final) = logit(p0) + sum_i w_i * [logit(p_i) - logit(ref_i)]

p0 is the prior: the stock's unconditional up-rate, or on a gap day
(|gap| >= 1% with enough similar gaps) the share of similar-gap days that
closed above the prior close. Each signal moves the logit by how far it sits
from *its own* no-information reference ref_i (the unconditional up-rate it
was measured against), times its weight. Without a gap, ref_i = p0 and this
is exactly the formula in DESIGN.md.

Guards: the result is capped to 35-65%, the range DESIGN.md sets for a
coin-flip-like prior. On a gap day the prior itself can sit far outside it
(after a -4% gap, closing above yesterday's close is often a 15% event), so
when p0 is outside 35-65% the cap becomes p0 ± 15 points instead: signals
still cannot move the forecast far from the prior, and an honest gap base
rate is not forced back toward 50%. An "agreement" count shows how many
weighted signals point each way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

CAP = (0.35, 0.65)
GAP_CAP_HALFWIDTH = 0.15


def cap_for(p0: float, cap: tuple[float, float] = CAP) -> tuple[float, float]:
    if cap[0] <= p0 <= cap[1]:
        return cap
    return (max(0.02, p0 - GAP_CAP_HALFWIDTH), min(0.98, p0 + GAP_CAP_HALFWIDTH))
ML_MAX_WEIGHT, ML_BSS_MULT = 0.6, 20.0
COND_MAX_WEIGHT = 0.5
NEWS_WEIGHT = 0.35
EVENT_PANEL_WEIGHT = 0.0       # until 20 scored panel events (M5)


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def ml_weight(best_bss: float | None, has_edge: bool) -> float:
    if not has_edge or best_bss is None:
        return 0.0
    return min(ML_MAX_WEIGHT, ML_BSS_MULT * max(0.0, best_bss))


def condition_weight(n: int) -> float:
    return min(COND_MAX_WEIGHT, 0.5 * math.sqrt(n / 100))


@dataclass
class Signal:
    name: str           # "ml", "news", "cond:<key>"
    label: str
    p: float
    ref: float
    weight: float
    detail: str = ""

    @property
    def contribution(self) -> float:
        return self.weight * (logit(self.p) - logit(self.ref))

    @property
    def direction(self) -> int:
        d = self.p - self.ref
        return 0 if abs(d) < 1e-9 else (1 if d > 0 else -1)


@dataclass
class Combined:
    p0: float
    p0_source: str              # "unconditional up-rate" or "gap base rate"
    p_raw: float                # before the cap
    p_final: float
    capped: bool
    signals: list[Signal] = field(default_factory=list)

    @property
    def agreement(self) -> tuple[int, int, int]:
        """(pointing up, pointing down, weighted signals)."""
        active = [s for s in self.signals if s.weight > 0]
        return (sum(s.direction > 0 for s in active), sum(s.direction < 0 for s in active), len(active))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["agreement"] = self.agreement
        for s, sd in zip(self.signals, d["signals"]):
            sd["contribution"] = s.contribution
        return d


def combine(p0: float, signals: list[Signal], p0_source: str = "unconditional up-rate",
            cap: tuple[float, float] = CAP) -> Combined:
    x = logit(p0) + sum(s.contribution for s in signals)
    raw = sigmoid(x)
    lo, hi = cap_for(p0, cap)
    final = min(max(raw, lo), hi)
    return Combined(p0=p0, p0_source=p0_source, p_raw=raw, p_final=final,
                    capped=final != raw, signals=signals)


def baserate_only(p_ref: float, signals: list[Signal]) -> float | None:
    """The conditional-base-rate signals combined alone (ledger column p_baserate)."""
    conds = [s for s in signals if s.name.startswith("cond:") and s.weight > 0]
    if not conds:
        return None
    return sigmoid(logit(p_ref) + sum(s.contribution for s in conds))
