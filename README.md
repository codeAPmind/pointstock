# PointStock — pre-market analysis (milestone M1)

Design: [DESIGN.md](DESIGN.md). M1 = data + features + GJR-GARCH(1,1)-t price bands + HTML/Markdown report.

## Setup (conda)

```bash
conda env create -f environment.yml   # creates env "premarket" (Python 3.11)
conda activate premarket
```

## Run

```bash
python -m premarket.run            # fetch (cached per close) and write reports/YYYY-MM-DD.html + .md
python -m premarket.run --offline  # cached data only; report shows a stale-data warning
python -m pytest                   # no-look-ahead + GARCH parameter-recovery tests
```

Tickers and settings live in `config.json`. Data cache: `data/prices/`, run manifests: `data/runs/<session>/`.
