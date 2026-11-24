"""Daily entry point:  python -m premarket.run [--offline] [--config PATH]

Pipeline (M1 + M2): prices (+ market context) -> feature table -> GJR-GARCH
bands -> out-of-sample 68% band coverage -> conditional and gap base rates
(event-day band widening) -> HTML + Markdown report.

What-if: --premarket NVDA=170.5 overrides the live pre-market quote.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import ROOT, load_config
from . import baserates as B
from . import data as D
from . import features as F
from . import garch as G
from . import models as M
from .refit import refit_ticker
from .report import TickerCard, run_meta_now, write_reports

log = logging.getLogger("premarket")


def next_session(last_bar: pd.Timestamp) -> pd.Timestamp:
    """Next weekday after the last completed bar (exchange holidays: M5)."""
    return (last_bar + pd.offsets.BDay(1)).normalize()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--offline", action="store_true", help="use cached data only")
    ap.add_argument("--premarket", action="append", default=[], metavar="TICKER=PRICE",
                    help="override the pre-market price (what-if / testing); repeatable")
    ap.add_argument("--reports-dir", default=None, help="write reports here instead of config paths.reports")
    args = ap.parse_args(argv)
    overrides = {k.upper(): float(v) for k, v in (o.split("=", 1) for o in args.premarket)}

    cfg = load_config(args.config)
    paths = cfg["paths"]
    Path(paths["logs"]).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(Path(paths["logs"]) / "run.log")])
    for noisy in ("yfinance", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    now = datetime.now(timezone.utc)
    manifest = D.RunManifest()
    years, gcfg = cfg["history_years"], cfg["garch"]

    market: dict[str, pd.DataFrame] = {}
    for name, sym in cfg["market_context"].items():
        try:
            market[name] = D.get_prices(sym, paths["data"], years, manifest, args.offline, now)
        except Exception as exc:
            log.warning("market context %s unavailable: %s", sym, exc)

    cards: list[TickerCard] = []
    last_bars: list[pd.Timestamp] = []
    for t in cfg["tickers"]:
        warnings: list[str] = []
        try:
            px = D.get_prices(t, paths["data"], years, manifest, args.offline, now)
        except Exception as exc:
            log.error("%s: no price data at all (%s); skipping", t, exc)
            continue
        status = next(s for s in reversed(manifest.sources) if s.name == f"prices:{t}")
        if status.stale:
            warnings.append(f"price data is stale ({status.detail})")
        last_bars.append(px.index[-1])

        table = F.build_features(px, market, sector=cfg.get("sector_for", {}).get(t))
        feat_dir = Path(paths["data"]) / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        table.to_parquet(feat_dir / f"{t}.parquet")

        fc = cov = None
        try:
            fc = G.forecast(px["Close"], gcfg["window"], gcfg["regime_high"], gcfg["regime_low"])
            cov = G.band_coverage(px["Close"], gcfg["window"], gcfg["coverage_test_days"])
            if fc.engine != "arch":
                warnings.append("arch fit failed; bands from the scipy fallback")
            if abs(cov.inside68 - 0.68) > 0.08:
                warnings.append(f"68% band coverage is {cov.inside68:.0%}, off target")
        except Exception as exc:
            log.exception("%s: GARCH failed", t)
            warnings.append(f"GARCH failed: {exc}")
        if len(px) < gcfg["window"] + 1:
            warnings.append(f"only {len(px) - 1} returns of history (window {gcfg['window']})")

        if t in overrides:
            pm = D.PreMarketQuote(t, overrides[t], now.isoformat(), "manual override")
            warnings.append(f"pre-market price is a manual override (${overrides[t]:,.2f})")
        else:
            pm = D.get_premarket(t, manifest, args.offline, now)
        if pm.price is None:
            warnings.append(f"pre-market price unavailable ({pm.detail})")

        # Base rates (M2)
        last_close = float(px["Close"].iloc[-1])
        conds = B.conditional_rates(px, table)
        gap = pm.price / last_close - 1 if pm.price is not None else None
        gr = B.gap_rates(px, gap, fc.sigma_next / 100 if fc else None) if gap is not None else None
        card = TickerCard(ticker=t, last_close=last_close,
                          last_date=str(px.index[-1].date()), forecast=fc, coverage=cov,
                          premarket=pm.price, premarket_asof=pm.asof_utc, warnings=warnings,
                          prior=B.unconditional_rate(px), conditions=conds, gap_rate=gr,
                          premarket_source=pm.source)
        # Direction models (M3): saved weekly bundle, refit inline if missing/outdated
        try:
            bundle = M.load_bundle(paths["data"], t)
            if bundle is None:
                log.info("%s: no saved models, running walk-forward refit now", t)
                bundle = refit_ticker(t, cfg, args.offline, market)
            card.direction = M.predict_today(bundle, table)
            warnings.extend(card.direction.warnings)
            if card.direction.p_ml is None:
                warnings.append("no direction edge: every model's walk-forward BSS ≤ 0")
            elif card.direction.best.bss < 0.01:
                warnings.append(f"direction edge is weak (best BSS {card.direction.best.bss:+.4f} < 0.01)")
        except Exception as exc:
            log.exception("%s: direction models failed", t)
            warnings.append(f"direction models failed: {exc}")
        if fc and card.event_day:
            card.event_bands = B.widened_bands(last_close, fc.band68, fc.band95, gr)
        thin = [r.label for r in conds if not r.usable]
        if thin:
            warnings.append(f"too few base-rate cases (N < {B.MIN_CASES}): " + "; ".join(thin))
        if gr is not None and not gr.usable:
            warnings.append(f"only {gr.n} similar past gaps (N < {B.MIN_CASES})")
        cards.append(card)

        if fc and cov:
            print(f"[{t}] close {fc.last_close:.2f} ({px.index[-1].date()}) | sigma {fc.sigma_next:.2f}% "
                  f"{fc.regime} | 68% {fc.band68[0]:.2f}-{fc.band68[1]:.2f} | "
                  f"95% {fc.band95[0]:.2f}-{fc.band95[1]:.2f} | 68% band coverage "
                  f"{cov.inside68:.1%} over {cov.n_days}d (95%: {cov.inside95:.1%})")
        print(f"[{t}] prior P(up) {card.prior[0]:.1%} (N={card.prior[1]}) | active conditions: "
              + (", ".join(f"{r.key} N={r.n} up {r.p_up_1d:.0%}" for r in conds) or "none"))
        if card.direction:
            d = card.direction
            print(f"[{t}] ML " + (f"P(up) {d.p_ml:.1%}" if d.p_ml is not None else "no edge")
                  + f" | best {d.best.model} BSS {d.best.bss:+.4f} | walk-forward {d.test_start}→{d.test_end}")
            print("\n".join("      " + l for l in M.format_table(d).splitlines()))
        if gr is not None:
            print(f"[{t}] gap {gr.gap:+.2%}: N={gr.n} close>open {gr.p_close_above_open:.0%} "
                  f"fills {gr.p_gap_fills:.0%} mean o->c {gr.mean_open_to_close:+.2%}")

    if not cards:
        log.error("no tickers produced a card")
        return 1

    session = next_session(max(last_bars)).date().isoformat()
    global_warnings = []
    stale = [s.name for s in manifest.sources if s.name.startswith("prices:") and (s.stale or not s.ok)]
    if stale:
        global_warnings.append("built on cached/stale data for: " + ", ".join(stale))
    meta = run_meta_now(session, cfg["timezones"]["local"], global_warnings)
    try:
        priors = B.load_event_priors(ROOT / cfg.get("event_priors", "event_base_rates.json"))
        n_cur = sum(p.usable for p in priors.values())
        meta["event_priors"] = (f"Event priors: {len(priors)} types loaded, {n_cur} curated. "
                                "Placeholders are shown for reference and never enter a probability; "
                                "headline detection starts with the news module (M4).")
    except Exception as exc:
        global_warnings.append(f"event_base_rates.json unreadable: {exc}")

    run_dir = Path(paths["data"]) / "runs" / session
    manifest.write(run_dir / "run_manifest.json")
    (run_dir / "forecasts.json").write_text(json.dumps(
        {c.ticker: {"garch": c.forecast.to_dict() if c.forecast else None,
                    "premarket": c.premarket, "prior": c.prior,
                    "conditions": [B.to_dict(r) for r in c.conditions],
                    "gap": B.to_dict(c.gap_rate) if c.gap_rate else None,
                    "event_bands": c.event_bands,
                    "direction": c.direction.to_dict() if c.direction else None} for c in cards}, indent=2, default=str))

    html_path, md_path = write_reports(cards, meta, args.reports_dir or paths["reports"])
    print(f"report: {html_path}\nsummary: {md_path}\nmanifest: {run_dir / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
