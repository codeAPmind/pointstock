# PointStock — pre-market analysis (milestones M1–M5)

Design: [DESIGN.md](DESIGN.md).

- **M1:** data + features + GJR-GARCH(1,1)-t price bands + HTML/Markdown report.
- **M2:** conditional base rates (with N), gap base rates on |gap| ≥ 1%, event-day band widening, curated event priors in `event_base_rates.json` (placeholders until you fill them).
- **M3:** four direction models (logistic regression, random forest, RBF SVM, naive Bayes) scored by walk-forward validation (21-day windows over the last ~750 days) with Brier skill vs a training-up-rate baseline; ensemble weights ∝ max(0, BSS), else "no direction edge".
- **M4:** news tier 1 (Yahoo + Google News, last 36h, relevance filter, LLM scoring (DeepSeek by default, Claude optional) with a keyword-lexicon fallback), Bayesian log-odds combiner → final P(up) with breakdown and agreement, forecast ledger (`data/forecast_ledger.csv`) with next-morning outcome filling and 60-day Brier skill scoring.
- **M5:** tier-2 analyst panel for materiality-3 events (actor map, scenario tree, five independent personas (DeepSeek by default), log-odds aggregate, disagreement score, expected value, tripwires that re-run it; history in `data/events/`), NYSE-calendar scheduling with cron, and push to Feishu / Telegram / email.

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
python -m premarket.run --premarket TSLA=401 --no-ledger --reports-dir /tmp/whatif   # what-if gap
python jobs/weekly_refit.py        # Sundays: walk-forward table + refit, saved to data/models/
python -m pytest                   # look-ahead, GARCH recovery, base-rate and walk-forward tests
```

Tickers and settings live in `config.json`. Data cache: `data/prices/`, models: `data/models/`, run manifests: `data/runs/<session>/`. The morning run only loads saved models; if none exist it runs the refit first (~10 s per ticker).

## Macro and industry context

Each card has a MACRO · INDUSTRY section (`premarket/macro.py`), built on the findings of
`../MacroEventQuntify`:

- **Futures since the last close** (S&P, Nasdaq, Russell, 10Y note, oil, gold, dollar) and the stock's
  pre-market gap split into market (SPY beta), sector (sector-ETF beta beyond the market) and
  stock-specific parts.
- **Today's macro calendar**: CPI, jobs, PPI, PCE, retail sales, GDP, claims and ISM from the FMP economic
  calendar (needs `FMP_API_KEY` in `.env`), FOMC days (`config.json` → `macro.fomc_dates`, update each
  January) and Treasury Note/Bond auctions (fiscaldata, no key). Released surprises are shown as z,
  signed hawkish-positive.
- **Event-day bands**: the band widens by √(variance ratio) of this stock's past moves on the same
  release type (release dates from MacroEventQuntify's `events.parquet`; only with N ≥ 20; capped ×2).
- **Macro and industry headlines** (`macro.headline_queries` + `industry.<TICKER>.queries`), scored for
  the stock's exposures by the LLM → signal `p_macro`, weight 0.2, tracked in the ledger and cut to 0
  automatically if its 60-day BSS is negative.

Macro surprises do not move P(up). MacroEventQuntify's walk-forward found they rank the opening gap
(IC ≈ 0.05) but not returns after the open (IC ≈ 0.005), so they widen bands and are shown as context.

## LLM provider (headline scoring and analyst panel)

`config.json` → `"llm": {"provider": "deepseek"}` (or `"anthropic"`, or `"auto"`). DeepSeek settings
live in `.env`: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL` (optional
`DEEPSEEK_PANEL_MODEL` for the panel). Calls use DeepSeek's JSON mode with the schema in the prompt;
every reply is validated and clipped. Without a key, or if a call fails, headlines fall back to the
keyword lexicon and the panel is skipped, and the report says so. `--no-llm` forces the lexicon.

## Combiner cap

DESIGN.md caps P(up) to 35–65%. When the prior itself is outside that range (a gap-day base rate,
e.g. 14% of similar −4% gaps closed above the prior close), the cap becomes prior ± 15 points so an
honest gap base rate is not pulled back toward 50%.

## Unattended runs (M5)

`scripts/run_premarket.sh` runs `python -m premarket.run --scheduled`: it skips NYSE holidays and
weekends, runs only between 08:15 and 09:25 ET, once per session (stamp in `data/runs/<date>/DONE`),
then pushes the summary. Email retries a dropped SMTP connection up to 3 times (auth errors
are not retried). Each result is appended to `logs/notify.jsonl` and the latest per-channel
status is kept in `data/runs/<date>/notify.json`. If a channel failed, the next scheduled
wake-up inside the window resends only that channel; `python -m premarket.run --resend [YYYY-MM-DD]`
does the same on demand. The machine clock is Singapore time, so cron fires at both 20:30 and 21:30
(08:30 ET in summer and winter) with :45 retries; see `scripts/crontab.txt`. Logs: `logs/cron.log`.

```bash
crontab -l > /tmp/crontab.bak && (cat /tmp/crontab.bak; cat scripts/crontab.txt) | crontab -
```

Push channels and the API key go in `PointStock/.env` (git-ignored, loaded by the script):

```
DEEPSEEK_API_KEY=...
FMP_API_KEY=...          # optional: macro calendar with consensus and actuals
PREMARKET_FEISHU_WEBHOOK=https://open.feishu.cn/open-apis/bot/v2/hook/...
# or PREMARKET_TELEGRAM_BOT_TOKEN / PREMARKET_TELEGRAM_CHAT_ID
# or PREMARKET_SMTP_HOST / _PORT / _USER / _PASSWORD, PREMARKET_EMAIL_FROM / _TO
```

The analyst panel makes 7 LLM calls per triggered event (only on materiality-3 news or a tripwire hit). `--no-panel` turns it off. Its weight in
P(up) is 0 until 20 panel events have been scored.
