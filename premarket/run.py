"""Daily entry point:  python -m premarket.run [--offline] [--config PATH]

Pipeline: score yesterday's ledger rows -> prices (+ market context) ->
feature table -> GJR-GARCH bands + coverage -> base rates (conditional, gap,
event-day widening) -> saved direction models -> news tier 1 -> Bayesian
combiner -> ledger row -> HTML + Markdown report.

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
from . import combine as C
from . import ledger as L
from . import models as M
from . import news as N
from .refit import refit_ticker
from .report import TickerCard, run_meta_now, write_reports

log = logging.getLogger("premarket")


def next_session(last_bar: pd.Timestamp) -> pd.Timestamp:
    """Next weekday after the last completed bar (exchange holidays: M5)."""
    return (last_bar + pd.offsets.BDay(1)).normalize()


def combine_card(c: TickerCard, disabled: list[str]) -> C.Combined:
    """Prior + weighted signals -> P(up), with weights from DESIGN.md's table."""
    p_uncond = c.prior[0]
    p0, src = p_uncond, "unconditional up-rate"
    gr = c.gap_rate
    if gr is not None and gr.usable and gr.p_up_vs_prev == gr.p_up_vs_prev:
        p0, src = gr.p_up_vs_prev, f"gap base rate, N = {gr.n}"
    sig: list[C.Signal] = []
    d = c.direction
    if d is not None:
        has_edge = d.p_ml is not None
        w = 0.0 if "p_ml" in disabled else C.ml_weight(d.best.bss, has_edge)
        sig.append(C.Signal("ml", "ML ensemble", d.p_ml if has_edge else d.up_rate, d.up_rate, w,
                            f"best {d.best.model} BSS {d.best.bss:+.4f}" if has_edge else "no edge (BSS ≤ 0)"))
    for r in c.conditions:
        if not r.usable:
            continue
        w = 0.0 if "p_baserate" in disabled else C.condition_weight(r.n)
        sig.append(C.Signal(f"cond:{r.key}", f"Base rate “{r.label}”", r.p_up_1d, p_uncond, w, f"N = {r.n}"))
    if c.news is not None and c.news.score is not None:
        p_news = N.news_probability(c.news.score, p_uncond)
        w = 0.0 if "p_news" in disabled else C.NEWS_WEIGHT
        sig.append(C.Signal("news", "News score", p_news, p_uncond, w,
                            f"score {c.news.score:+.2f}, {c.news.n_relevant} headlines, {c.news.scorer}"))
    return C.combine(p0, sig, src)


def ledger_row(c: TickerCard, session: str, now: datetime) -> dict:
    cb, f = c.combined, c.forecast
    b68 = c.event_bands[0] if c.event_bands else (f.band68 if f else (None, None))
    b95 = c.event_bands[1] if c.event_bands else (f.band95 if f else (None, None))
    ml = next((s for s in cb.signals if s.name == "ml"), None)
    nw = next((s for s in cb.signals if s.name == "news"), None)
    conds = [s for s in cb.signals if s.name.startswith("cond:")]
    return {
        "date": session, "ticker": c.ticker, "last_close_date": c.last_date, "last_close": c.last_close,
        "premarket": c.premarket, "gap": c.gap, "p_uncond": c.prior[0], "p_prior": cb.p0,
        "p_ml": c.direction.p_ml if c.direction else None,
        "p_news": nw.p if nw else None, "p_baserate": C.baserate_only(c.prior[0], cb.signals),
        "p_final": cb.p_final, "w_ml": ml.weight if ml else 0.0, "w_news": nw.weight if nw else 0.0,
        "w_baserate": sum(s.weight for s in conds), "sigma_pred": f.sigma_next / 100 if f else None,
        "band68_low": b68[0], "band68_high": b68[1], "band95_low": b95[0], "band95_high": b95[1],
        "active_conditions": ";".join(r.key for r in c.conditions),
        "event_types": ";".join(sorted(c.news.events)) if c.news else "",
        "news_scorer": c.news.scorer if c.news else "",
        "outcome_up": None, "actual_close": None, "actual_ret": None, "inside_band68": None,
        "model_version": c.direction.fitted_on if c.direction else "",
        "prompt_version": N.PROMPT_VERSION, "created_utc": now.isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--offline", action="store_true", help="use cached data only")
    ap.add_argument("--premarket", action="append", default=[], metavar="TICKER=PRICE",
                    help="override the pre-market price (what-if / testing); repeatable")
    ap.add_argument("--reports-dir", default=None, help="write reports here instead of config paths.reports")
    ap.add_argument("--no-llm", action="store_true", help="score headlines with the keyword lexicon only")
    ap.add_argument("--no-ledger", action="store_true", help="do not write to the forecast ledger (what-if runs)")
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

    priors = {}
    try:
        priors = B.load_event_priors(ROOT / cfg.get("event_priors", "event_base_rates.json"))
    except Exception as exc:
        log.warning("event_base_rates.json unreadable: %s", exc)
    ncfg = cfg.get("news", {})
    prices: dict[str, pd.DataFrame] = {}
    sessions: dict[str, str] = {}

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
        prices[t] = px
        sessions[t] = next_session(px.index[-1]).date().isoformat()

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
        # News tier 1 (M4)
        tn = ncfg.get(t, {"company": t, "names": [t], "query": f"{t} stock"})
        try:
            raw, failed = D.get_news(t, tn["query"], paths["data"], manifest, args.offline, now,
                                     ncfg.get("hours", 36))
            card.news = N.analyze(t, tn["company"], tn["names"], raw, failed, priors, paths["data"],
                                  sessions[t], use_llm=False if args.no_llm else None)
            warnings.extend(card.news.warnings)
            if card.news.n_relevant == 0:
                warnings.append("no relevant headlines in the last 36 hours")
        except Exception as exc:
            log.exception("%s: news failed", t)
            warnings.append(f"news module failed: {exc}")
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

    # Ledger: fill outcomes for closed sessions and score before using today's weights
    ledger = L.load(paths["data"])
    ledger, n_filled = L.fill_outcomes(ledger, prices)
    scorecard = L.score(ledger)
    if n_filled:
        print(f"ledger: filled outcomes for {n_filled} earlier row(s)")
    print(scorecard.summary())

    rows = []
    for c in cards:
        c.combined = combine_card(c, scorecard.disabled)
        rows.append(ledger_row(c, sessions[c.ticker], now))
        cb = c.combined
        print(f"[{c.ticker}] P(up) {cb.p_final:.1%} (prior {cb.p0:.1%} from {cb.p0_source}) | "
              + " · ".join(f"{s.name} {s.p:.0%} w{s.weight:.2f}" for s in cb.signals)
              + f" | agreement {cb.agreement[0]} up / {cb.agreement[1]} down of {cb.agreement[2]}")
    if not args.no_ledger:
        ledger = L.upsert(ledger, rows)
        L.save(ledger, paths["data"])

    session = next_session(max(last_bars)).date().isoformat()
    global_warnings = []
    stale = [s.name for s in manifest.sources if s.name.startswith("prices:") and (s.stale or not s.ok)]
    if stale:
        global_warnings.append("built on cached/stale data for: " + ", ".join(stale))
    meta = run_meta_now(session, cfg["timezones"]["local"], global_warnings)
    meta["ledger_summary"] = scorecard.summary() + (" (what-if run: not written to the ledger)" if args.no_ledger else "")
    if priors:
        n_cur = sum(p.usable for p in priors.values())
        meta["event_priors"] = (f"Event priors: {len(priors)} types loaded, {n_cur} curated. "
                                "Uncurated placeholders are shown for reference and never enter a probability.")
    else:
        global_warnings.append("event_base_rates.json unreadable")

    run_dir = Path(paths["data"]) / "runs" / session
    manifest.write(run_dir / "run_manifest.json")
    (run_dir / "forecasts.json").write_text(json.dumps(
        {c.ticker: {"garch": c.forecast.to_dict() if c.forecast else None,
                    "premarket": c.premarket, "prior": c.prior,
                    "conditions": [B.to_dict(r) for r in c.conditions],
                    "gap": B.to_dict(c.gap_rate) if c.gap_rate else None,
                    "event_bands": c.event_bands,
                    "direction": c.direction.to_dict() if c.direction else None,
                    "news": c.news.to_dict() if c.news else None,
                    "combined": c.combined.to_dict() if c.combined else None} for c in cards},
        indent=2, default=str))

    html_path, md_path = write_reports(cards, meta, args.reports_dir or paths["reports"])
    print(f"report: {html_path}\nsummary: {md_path}\nmanifest: {run_dir / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
