"""Weekly refit: walk-forward score the direction models, refit on all data, save.

    python -m premarket.refit [--offline] [--tickers NVDA TSLA]

Run on Sundays (DESIGN.md "When it runs"); the morning run only loads the
saved bundles. Writes data/models/<TICKER>.joblib plus a walk-forward
predictions CSV and a scores JSON next to it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import load_config
from . import data as D
from . import features as F
from . import models as M

log = logging.getLogger("premarket.refit")


def build_table(ticker: str, cfg: dict, offline: bool = False,
                market: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    paths, years = cfg["paths"], cfg["history_years"]
    now = datetime.now(timezone.utc)
    if market is None:
        market = {}
        for name, sym in cfg["market_context"].items():
            try:
                market[name] = D.get_prices(sym, paths["data"], years, None, offline, now)
            except Exception as exc:
                log.warning("market context %s unavailable: %s", sym, exc)
    px = D.get_prices(ticker, paths["data"], years, None, offline, now)
    return F.build_features(px, market, sector=cfg.get("sector_for", {}).get(ticker))


def refit_ticker(ticker: str, cfg: dict, offline: bool = False,
                 market: dict[str, pd.DataFrame] | None = None) -> M.ModelBundle:
    t0 = time.time()
    table = build_table(ticker, cfg, offline, market)
    bundle, wf = M.fit_bundle(table, ticker)
    path = M.save_bundle(bundle, cfg["paths"]["data"])
    wf.predictions.to_csv(path.with_name(f"{ticker}_walkforward.csv"))
    path.with_name(f"{ticker}_scores.json").write_text(json.dumps({
        "ticker": ticker, "fitted_on": bundle.fitted_on, "test": [wf.test_start, wf.test_end],
        "n_folds": len(wf.folds), "up_rate": bundle.up_rate, "weights": bundle.weights,
        "scores": {k: asdict(v) for k, v in wf.scores.items()}}, indent=2))
    print(f"\n[{ticker}] walk-forward {wf.test_start} → {wf.test_end}, {len(wf.folds)} folds of "
          f"{M.STEP} days, {len(bundle.feature_cols)} features, {time.time() - t0:.0f}s")
    print(M.format_table(wf))
    edge = "no direction edge (every BSS ≤ 0)" if sum(bundle.weights.values()) == 0 else \
        f"best {wf.best.model} BSS {wf.best.bss:+.4f}"
    print(f"→ {edge}")
    return bundle


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--tickers", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    Path(cfg["paths"]["logs"]).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(Path(cfg["paths"]["logs"]) / "refit.log")])
    failed = 0
    for t in args.tickers or cfg["tickers"]:
        try:
            refit_ticker(t, cfg, args.offline)
        except Exception:
            log.exception("refit failed for %s", t)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
