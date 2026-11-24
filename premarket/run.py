"""Daily entry point:  python -m premarket.run [--offline] [--config PATH]

M1 pipeline: prices (+ market context) -> feature table -> GJR-GARCH bands
-> out-of-sample 68% band coverage check -> HTML + Markdown report.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import load_config
from . import data as D
from . import features as F
from . import garch as G
from .report import TickerCard, run_meta_now, write_reports

log = logging.getLogger("premarket")


def next_session(last_bar: pd.Timestamp) -> pd.Timestamp:
    """Next weekday after the last completed bar (exchange holidays: M5)."""
    return (last_bar + pd.offsets.BDay(1)).normalize()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--offline", action="store_true", help="use cached data only")
    args = ap.parse_args(argv)

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

        pm = D.get_premarket(t, manifest, args.offline, now)
        if pm.price is None:
            warnings.append(f"pre-market price unavailable ({pm.detail})")

        cards.append(TickerCard(ticker=t, last_close=float(px["Close"].iloc[-1]),
                                last_date=str(px.index[-1].date()), forecast=fc, coverage=cov,
                                premarket=pm.price, premarket_asof=pm.asof_utc, warnings=warnings))

        if fc and cov:
            print(f"[{t}] close {fc.last_close:.2f} ({px.index[-1].date()}) | sigma {fc.sigma_next:.2f}% "
                  f"{fc.regime} | 68% {fc.band68[0]:.2f}-{fc.band68[1]:.2f} | "
                  f"95% {fc.band95[0]:.2f}-{fc.band95[1]:.2f} | 68% band coverage "
                  f"{cov.inside68:.1%} over {cov.n_days}d (95%: {cov.inside95:.1%})")

    if not cards:
        log.error("no tickers produced a card")
        return 1

    session = next_session(max(last_bars)).date().isoformat()
    global_warnings = []
    stale = [s.name for s in manifest.sources if s.name.startswith("prices:") and (s.stale or not s.ok)]
    if stale:
        global_warnings.append("built on cached/stale data for: " + ", ".join(stale))
    meta = run_meta_now(session, cfg["timezones"]["local"], global_warnings)

    run_dir = Path(paths["data"]) / "runs" / session
    manifest.write(run_dir / "run_manifest.json")
    (run_dir / "forecasts.json").write_text(json.dumps(
        {c.ticker: (c.forecast.to_dict() if c.forecast else None) for c in cards}, indent=2, default=str))

    html_path, md_path = write_reports(cards, meta, paths["reports"])
    print(f"report: {html_path}\nsummary: {md_path}\nmanifest: {run_dir / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
