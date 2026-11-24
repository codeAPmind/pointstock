# PointStock — pre-market analysis (milestones M1–M3)

Design: [DESIGN.md](DESIGN.md).

- **M1:** data + features + GJR-GARCH(1,1)-t price bands + HTML/Markdown report.
- **M2:** conditional base rates (with N), gap base rates on |gap| ≥ 1%, event-day band widening, curated event priors in `event_base_rates.json` (placeholders until you fill them).
- **M3:** four direction models (logistic regression, random forest, RBF SVM, naive Bayes) scored by walk-forward validation (21-day windows over the last ~750 days) with Brier skill vs a training-up-rate baseline; ensemble weights ∝ max(0, BSS), else "no direction edge".

## Setup (conda)

```bash
conda env create -f environment.yml   # creates env "premarket" (Python 3.11)
conda activate premarket
```

If `conda` is not found, run `~/miniconda3/bin/conda init zsh` once, or `source ~/miniconda3/bin/activate premarket`.

## Run

```bash
python -m premarket.run            # fetch (cached per close) and write reports/YYYY-MM-DD.html + .md
python -m premarket.run --offline  # cached data only; report shows a stale-data warning
python -m premarket.run --premarket TSLA=401 --reports-dir /tmp/whatif   # what-if gap
python jobs/weekly_refit.py        # Sundays: walk-forward table + refit, saved to data/models/
python -m pytest                   # look-ahead, GARCH recovery, base-rate and walk-forward tests
```

Tickers and settings live in `config.json`. Data cache: `data/prices/`, models: `data/models/`, run manifests: `data/runs/<session>/`. The morning run only loads saved models; if none exist it runs the refit first (~10 s per ticker).
