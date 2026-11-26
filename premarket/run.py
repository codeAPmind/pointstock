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
import math
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
from . import macro as MC
from . import models as M
from . import news as N
from . import notify as NT
from . import panel as P
from . import schedule as S
from .refit import refit_ticker
from .report import TickerCard, run_meta_now, write_reports

log = logging.getLogger("premarket")


def next_session(last_bar: pd.Timestamp) -> pd.Timestamp:
    """Next NYSE session after the last completed bar (weekday fallback)."""
    try:
        return pd.Timestamp(S._calendar().next_session(pd.Timestamp(last_bar).normalize())).tz_localize(None)
    except Exception:
        return (last_bar + pd.offsets.BDay(1)).normalize()


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def build_macro_context(t, tn, ind, mcfg, px, pm, spy_px, spy_gap, futures, fut_failed, cal_events,
                        cal_warn, surprises, release_dates, macro_raw, macro_failed, company_news,
                        provider, paths, years, manifest, offline, now) -> MC.MacroContext:
    """Gap split, event-day volatility, sensitivities and scored macro/industry headlines for one ticker."""
    warnings = list(cal_warn)
    if fut_failed:
        warnings.append("futures unavailable: " + ", ".join(fut_failed))
    last_close = float(px["Close"].iloc[-1])
    # Gap split into market / sector / stock-specific
    split, sec_sym, sec_gap = None, ind.get("etf"), None
    if pm.price is not None and spy_gap is not None and spy_px is not None:
        sec_px = None
        if sec_sym:
            try:
                sec_px = D.get_prices(sec_sym, paths["data"], years, manifest, offline, now)
                sq = D.get_premarket(sec_sym, manifest, offline, now)
                if sq.price:
                    sec_gap = math.log(sq.price / float(sec_px["Close"].iloc[-1]))
            except Exception as exc:
                warnings.append(f"sector ETF {sec_sym} unavailable ({type(exc).__name__})")
        split = MC.gap_decomposition(px["Close"], spy_px["Close"], sec_px["Close"] if sec_px is not None else None,
                                     math.log(pm.price / last_close), spy_gap, sec_gap)
    # Event-day volatility
    asof = pd.Timestamp(S.session_date(now))
    mult, details = MC.vol_multiplier(px["Close"], cal_events, release_dates, asof)
    # Headlines: shared macro + this ticker's industry queries, minus its own company headlines
    ind_raw, ind_failed = MC.fetch_headlines(ind.get("queries", []), f"{t}_IND", paths["data"], now,
                                             mcfg.get("hours", 24), offline)
    own = {N._key(h.title) for h in company_news.headlines} if company_news else set()
    pool = [h for h in MC.dedupe(macro_raw + ind_raw) if N._key(h["title"]) not in own
            and not N.is_relevant(h["title"], t, tn.get("names", [t]))]
    if macro_failed or ind_failed:
        warnings.append("some macro/industry headline queries failed (cache used where available)")
    mnews = MC.score_macro_news(t, tn.get("company", t), ind.get("exposure", "the US stock market"), pool, provider)
    warnings += mnews.warnings
    return MC.MacroContext(futures=futures, events=cal_events, surprises=surprises, gap_split=split,
                           sector_symbol=sec_sym, sector_gap=sec_gap, vol_mult=mult, vol_details=details,
                           sensitivity=MC.sensitivity(_resolve(mcfg.get("sensitivity_parquet")), t),
                           news=mnews, warnings=warnings)


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
    mn = c.macro.news if c.macro is not None else None
    if mn is not None and mn.score is not None:
        p_macro = MC.macro_probability(mn.score, p_uncond)
        w = 0.0 if "p_macro" in disabled else MC.MACRO_WEIGHT
        sig.append(C.Signal("macro", "Macro & industry news", p_macro, p_uncond, w,
                            f"score {mn.score:+.2f}, {len(mn.headlines)} headlines, {mn.scorer}"))
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
    mc = next((s for s in cb.signals if s.name == "macro"), None)
    conds = [s for s in cb.signals if s.name.startswith("cond:")]
    return {
        "date": session, "ticker": c.ticker, "last_close_date": c.last_date, "last_close": c.last_close,
        "premarket": c.premarket, "gap": c.gap, "p_uncond": c.prior[0], "p_prior": cb.p0,
        "p_ml": c.direction.p_ml if c.direction else None,
        "p_news": nw.p if nw else None, "p_baserate": C.baserate_only(c.prior[0], cb.signals),
        "p_macro": mc.p if mc else None, "w_macro": mc.weight if mc else 0.0,
        "macro_events": ";".join(sorted({e.key for e in c.macro.events})) if c.macro else "",
        "macro_vol_mult": c.macro.vol_mult if c.macro else 1.0,
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
    ap.add_argument("--no-panel", action="store_true", help="skip the tier-2 analyst panel (no Opus calls)")
    ap.add_argument("--scheduled", action="store_true",
                    help="unattended mode: skip holidays, run once per session inside the pre-market window, push")
    ap.add_argument("--force", action="store_true", help="with --scheduled: ignore the time window and run stamp")
    ap.add_argument("--notify", action="store_true", help="push the summary to configured channels")
    ap.add_argument("--resend", nargs="?", const="today", default=None, metavar="SESSION",
                    help="re-push a saved report (failed channels only) without recomputing; "
                         "default is today's NY session")
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
    if args.resend is not None:
        session = S.session_date(now) if args.resend == "today" else args.resend
        try:
            res = NT.resend(session, paths["data"], args.reports_dir or paths["reports"], paths["logs"])
        except FileNotFoundError as exc:
            print(f"resend: {exc}")
            return 1
        print("push: " + (", ".join(f"{k} {v}" for k, v in res.items()) or "nothing to resend"))
        return 0
    if args.scheduled:
        go, reason = S.should_run(paths["data"], now, force=args.force)
        print(f"[scheduled] {reason}")
        if not go:
            if "already ran" in reason:
                session = S.session_date(now)
                pending = NT.failed_channels(paths["data"], session)
                if pending:
                    print(f"[scheduled] resend {', '.join(pending)} for {session}")
                    res = NT.resend(session, paths["data"], args.reports_dir or paths["reports"], paths["logs"])
                    print("push: " + (", ".join(f"{k} {v}" for k, v in res.items()) or "nothing to resend"))
            return 0
    manifest = D.RunManifest()
    years, gcfg = cfg["history_years"], cfg["garch"]

    priors = {}
    try:
        priors = B.load_event_priors(ROOT / cfg.get("event_priors", "event_base_rates.json"))
    except Exception as exc:
        log.warning("event_base_rates.json unreadable: %s", exc)
    ncfg = cfg.get("news", {})
    provider = None if args.no_llm else N.LLM.active_provider(cfg.get("llm", {}).get("provider", "auto"))
    print(f"LLM provider: {provider or 'none (keyword lexicon, no analyst panel)'}")
    prices: dict[str, pd.DataFrame] = {}
    sessions: dict[str, str] = {}

    market: dict[str, pd.DataFrame] = {}
    for name, sym in cfg["market_context"].items():
        try:
            market[name] = D.get_prices(sym, paths["data"], years, manifest, args.offline, now)
        except Exception as exc:
            log.warning("market context %s unavailable: %s", sym, exc)

    # Macro context shared by all tickers: futures, today's calendar, macro headlines
    mcfg, icfg = cfg.get("macro", {}), cfg.get("industry", {})
    day_ny = S.session_date(now)
    spy_px = market.get("SPY")
    prev_close_utc = (datetime.combine(spy_px.index[-1].date(), datetime.min.time().replace(hour=16),
                                       tzinfo=MC.NY).astimezone(timezone.utc) if spy_px is not None else now)
    futures, fut_failed = MC.futures_moves(mcfg.get("futures", {}), prev_close_utc, now, args.offline)
    cal_events, cal_warn = MC.todays_calendar(day_ny, mcfg, paths["data"], args.offline)
    release_dates = MC.load_release_dates(_resolve(mcfg.get("release_dates_parquet")))
    macro_raw, macro_failed = MC.fetch_headlines(mcfg.get("headline_queries", []), "MACRO", paths["data"],
                                                 now, mcfg.get("hours", 24), args.offline)
    spy_pm = None if args.offline else D.get_premarket("SPY", manifest, False, now)
    spy_gap = (math.log(spy_pm.price / float(spy_px["Close"].iloc[-1]))
               if spy_pm and spy_pm.price and spy_px is not None else None)
    surprises = MC.group_surprise(cal_events)
    print(f"macro: {len(cal_events)} calendar event(s) today "
          f"[{', '.join(sorted({e.key for e in cal_events})) or 'none'}], {len(macro_raw)} macro headlines, "
          f"SPY pre-market {'n/a' if spy_gap is None else f'{spy_gap:+.2%}'}")

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
        tech = F.technical_read(table, float(px["Close"].iloc[-1]), str(px.index[-1].date()))
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
                          premarket_source=pm.source, technical=tech)
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
                                  sessions[t], provider=provider)
            warnings.extend(card.news.warnings)
            if card.news.n_relevant == 0:
                warnings.append("no relevant headlines in the last 36 hours")
        except Exception as exc:
            log.exception("%s: news failed", t)
            warnings.append(f"news module failed: {exc}")
        # Tier-2 analyst panel for materiality-3 events (M5); display only, weight 0
        if card.news is not None and not args.no_panel and cfg.get("panel", {}).get("enabled", True):
            try:
                card.panels, pw = P.process_events(
                    t, tn["company"], card.news, priors, pm.price or float(px["Close"].iloc[-1]),
                    paths["data"], sessions[t], llm_ok=provider)
                warnings.extend(pw)
            except Exception as exc:
                log.exception("%s: panel failed", t)
                warnings.append(f"analyst panel failed: {exc}")
        # Macro and industry context (M4b)
        try:
            card.macro = build_macro_context(
                t, tn, icfg.get(t, {}), mcfg, px, pm, spy_px, spy_gap, futures, fut_failed, cal_events,
                cal_warn, surprises, release_dates, macro_raw, macro_failed, card.news, provider,
                paths, years, manifest, args.offline, now)
            warnings.extend(card.macro.warnings)
        except Exception as exc:
            log.exception("%s: macro context failed", t)
            warnings.append(f"macro context failed: {exc}")
        if fc and card.event_day:
            card.event_bands = B.widened_bands(last_close, fc.band68, fc.band95, gr)
        if fc and card.macro is not None and card.macro.vol_mult > 1:
            m = card.macro.vol_mult
            b68, b95 = card.event_bands or (fc.band68, fc.band95)
            w68, w95 = MC.widen(last_close, fc.band68, m), MC.widen(last_close, fc.band95, m)
            card.event_bands = ((min(b68[0], w68[0]), max(b68[1], w68[1])),
                                (min(b95[0], w95[0]), max(b95[1], w95[1])))
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
        if tech:
            def _signed(x, spec="+.1%"):
                return "n/a" if x is None else format(x, spec)
            print(f"[{t}] technical MA20 {_signed(tech.dist_ma20)} MA50 {_signed(tech.dist_ma50)} "
                  f"MA200 {_signed(tech.dist_ma200)} | RSI {_signed(tech.rsi14, '.1f')} | "
                  f"MACD hist {_signed(tech.macd_hist_pct, '+.2%')} | "
                  f"tally {tech.checks_up}/{tech.checks_n} | stack {tech.stack}")
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
                    "combined": c.combined.to_dict() if c.combined else None,
                    "panels": [{"event_id": o.record.event_id, "ran_today": o.ran_today, "note": o.note}
                               for o in c.panels]} for c in cards},
        indent=2, default=str))

    html_path, md_path = write_reports(cards, meta, args.reports_dir or paths["reports"])
    print(f"report: {html_path}\nsummary: {md_path}\nmanifest: {run_dir / 'run_manifest.json'}")
    if args.notify or args.scheduled:
        res = NT.push(cards, meta, html_path, data_dir=paths["data"], logs_dir=paths["logs"])
        print("push: " + (", ".join(f"{k} {v}" for k, v in res.items()) or "no channel configured"))
    if args.scheduled:
        S.mark_done(paths["data"], session, f"{len(cards)} tickers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
