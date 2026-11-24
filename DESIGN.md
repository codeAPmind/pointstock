# Pre-Market Stock Analysis System — Design

Nov 24, 2026 · @milen

## Goal and scope

Every US trading day at 8:30 a.m. New York time (one hour before the open), the system writes one report per stock: today's expected price range, the probability of an up day, the news and events that matter, and how much to trust each number.

**Tickers (v1):** TSLA, NVDA, OPEN (Opendoor). The list lives in `config.json`, so adding a ticker is one line.

**What it predicts, ranked by how reliable it will be:**

1. **Volatility (the size of today's move)** — the strongest signal. Volatility clusters, so GARCH can forecast tomorrow's range well. Output: a 68% and a 95% price band.
2. **Event priors** — when the news shows a known event type (CEO change, merger talks, probe), the report gives the historical base rate for what usually happens next.
3. **Direction (up or down)** — the weakest signal. For large, liquid stocks, daily direction is close to a coin flip. ML models will usually reach 50–53% accuracy out-of-sample. The system *measures* any edge instead of assuming one, and falls back to the base rate when a model shows no skill.

**Out of scope for v1:** placing trades, intraday (minute-bar) prediction, and portfolio sizing. The report supports a decision; it does not make one.

## Architecture

The system is one Python package run by a scheduler. Data flows top to bottom; the only loop is the ledger, which scores yesterday's forecasts and adjusts how much each engine is trusted.

&#91;embedded content: daily pipeline · 4 sources, 4 engines, 1 combiner\]

GARCH sends its price bands straight to the report; the other three engines produce probabilities that the combiner merges into one P(up).

**Module layout**

```
premarket/
  config.json              tickers, paths, weights
  event_base_rates.json    curated event priors
  premarket/
    data.py                prices, pre-market quote, news, filings + cache
    features.py            feature table (no look-ahead)
    garch.py               GJR-GARCH fit, sigma, price bands
    models.py              4 classifiers, walk-forward, BSS
    baserates.py           conditional, gap, event detection
    news.py                LLM headline scoring (tier 1)
    panel.py               actor map, scenarios, personas (tier 2)
    combine.py             log-odds combiner
    ledger.py              append forecasts, fill outcomes, score
    report.py              HTML + Markdown output
    run.py                 daily entry point
  jobs/weekly_refit.py     refit models every Sunday
  tests/
  data/  reports/  logs/
```

## Data layer

All v1 sources are free and need no API key. Every fetch caches to disk, so a failed source falls back to yesterday's cache instead of crashing the run.

| Data | Source (v1) | Upgrade path | Stored as |
| --- | --- | --- | --- |
| Daily k-line (OHLCV), 6 years, split-adjusted | `yfinance` | Polygon.io, Alpaca, Tiingo | `data/prices/<TICKER>.parquet` |
| Pre-market price (gap vs yesterday's close) | `yfinance` `fast_info` | Polygon snapshot | in the run record |
| News headlines, last 36 hours | Yahoo via `yfinance` + Google News RSS | Benzinga, Finnhub, NewsAPI | `data/news/<date>_<TICKER>.json` |
| Company filings (8-K, 13D, DEF 14A) | SEC EDGAR full-text search | — | `data/filings/` |
| Options implied volatility | — (v2) | `yfinance` option chains, CBOE | `data/options/` |
| Prediction-market odds for named events | — (v2) | Polymarket public API | `data/markets/` |
| Market context (SPY, QQQ, VIX, SOXX) | `yfinance` | — | same as k-line |

**Rules for the data layer**

- Timestamps in UTC internally; the report converts to New York and Singapore time.
- Every feature at day *t* uses only data known at the close of day *t* (no look-ahead). A unit test checks this.
- Headlines are de-duplicated on the first 80 characters, lower-cased.
- Each run writes a `run_manifest.json` listing which sources succeeded, so a report built on stale data says so at the top.

## Quant engine

Three parts: a feature table built from the k-line, a GARCH model for the size of the move, and four classifiers for its direction, all scored by walk-forward validation.

### Features (one row per day)

| Group | Features |
| --- | --- |
| Returns | 1-, 5-, 20-day log returns; overnight gap (open vs prior close); open-to-close |
| Trend | distance from 20/50/200-day moving averages; MACD histogram / price |
| Mean reversion | RSI(14); Bollinger z-score (20 days); up/down streak length |
| Volatility | ATR(14) / price; realized vol over 5 and 20 days; their ratio; day range / close |
| Volume | log-volume z-score vs 20 days |
| Candle shape | where the close sits in the day's high–low range (0 = low, 1 = high) |
| Market context | SPY and QQQ 1-day return; VIX level and 5-day change; SOXX for NVDA |
| News (from the news module) | daily sentiment score; headline count z-score; event-type flags |

Target: `target_up = 1` when the next close is above today's close. Also store `next_ret` (log return) for range and base-rate work.

### GARCH: the size of the move

Model: GJR-GARCH(1,1) with Student-t errors on daily log returns, fit on the last 1,000 days. GJR adds the leverage effect (down days raise volatility more than up days); Student-t handles fat tails.

```latex
\sigma_t^2 = \omega + (\alpha + \gamma \, I_{\epsilon_{t-1}<0}) \, \epsilon_{t-1}^2 + \beta \, \sigma_{t-1}^2
```

- Library: `arch` (`arch_model(r*100, vol='GARCH', p=1, o=1, q=1, dist='t')`). Keep a small scipy fallback with the same likelihood.
- Outputs: next-day sigma; 68% and 95% price bands from Student-t quantiles; regime label (HIGH if sigma > 1.3 × long-run sigma, LOW if < 0.75 ×).
- Validation: the share of days that close inside the 68% band should be close to 68%. Show it in the report.
- Pre-market adjustment: if the pre-market gap is already larger than 1 sigma, the report flags "event day" and widens the band using the gap base rates below.

### Direction models

| Model | Settings (start here) | Why it's in the panel |
| --- | --- | --- |
| Logistic regression | standardized features, C = 0.1 | simple, stable baseline |
| Random forest | 300 trees, max depth 4, min leaf 40 | non-linear effects; feature importance |
| SVM (RBF) | C = 0.5, Platt-scaled probabilities | different decision boundary |
| Gaussian naive Bayes | standardized features | the Bayesian baseline; fast |
| Gradient boosting (v2) | LightGBM, shallow trees | usually best on tabular data |

Depth and leaf sizes are kept small on purpose: about 1,500 noisy daily rows overfit easily.

### Walk-forward validation (the honesty check)

1. Train on all days before the test window, predict the next 21 trading days, then slide forward and retrain.
2. Test on the last \~750 days (about 3 years) so every model is scored only on data it never saw.
3. Compare each model against a no-skill baseline that always predicts the historical up-rate.
4. Score with the Brier score and Brier skill score: BSS = 1 − Brier(model) / Brier(baseline). BSS ≤ 0 means no edge.
5. A model's weight in the ensemble is proportional to max(0, BSS). If every model has BSS ≤ 0, the report says "no direction edge" and uses the base rate.

Never use random K-fold cross-validation on time series: it leaks future data into training and makes models look far better than they are.

## Base rates

Base rates answer one question: "the last N times this stock was in today's situation, what happened next?" They are the prior that every other signal adjusts. There are three kinds.

### 1. Conditional base rates (computed from the stock's own history)

Each morning, check which conditions are true at yesterday's close. For each one, report N, P(up next day), mean and median next-day return, and P(up over 5 days).

| Condition | Rule |
| --- | --- |
| Big up / down day | yesterday's return > +2 or < −2 × 60-day sigma |
| Extreme day | yesterday's move > +15% or < −15% (e.g. NVDA on 2026-11-23) |
| Streak | 3+ up closes or 3+ down closes in a row |
| Overbought / oversold | RSI(14) > 70 or < 30 |
| Trend regime | above or below the 200-day MA |
| Volume spike | volume z-score > 2 |
| Close position | close in the top or bottom 10% of the day's range |
| Drawdown | down more than 30% from the 60-day high |

Only use a condition with at least 25 past cases; show N beside every number so a 70% from 12 cases is not read like a 70% from 400.

### 2. Gap base rates (uses the pre-market price)

Gap = pre-market price / yesterday's close − 1. If |gap| ≥ 1%, find past days with a similar gap (0.6× to 1.6× the size, same sign) and report: P(close above open), P(gap fills the same day), and mean open-to-close return. This is often the most useful line in the report on a news day.

### 3. Corporate event priors (a table you curate)

A JSON file, `event_base_rates.json`, holds one entry per event type: the keywords that detect it, the question it answers, the prior, and the past cases behind it.

| Event type | Example question | Example cases |
| --- | --- | --- |
| CEO departure | Does the stock recover the event-day drop within 30 trading days? | NVDA/Huang resignation 2026-11-12 |
| CEO return | Does the return rally hold after 20 trading days? | OpenAI/Altman 2023; NVDA/Huang 2026-11-23 |
| Merger talks | Do talks reach a signed agreement within 6 months? | SpaceX/xAI 2026; Tesla/SolarCity 2016 |
| Regulatory / legal | Does a probe lead to a material penalty within 12 months? | — |
| Earnings | computed from the stock's own past earnings-day moves | — |
| Analyst action | Does an upgrade/downgrade drift continue the next day? | — |

The priors start as placeholders. Fill each one by counting real historical cases; this table is where your own research compounds over time. The system displays these priors but never feeds an uncurated placeholder into the final probability.

## News and event analysis

The news module turns headlines into three outputs: a sentiment score, event flags, and, for big events only, a structured scenario analysis by an LLM analyst panel.

### Tier 1: every headline, every day (cheap)

1. **Relevance filter** — drop headlines that only mention the ticker in a list ("10 stocks to watch").
2. **Sentiment** — an LLM call (Claude Haiku or Sonnet via the Anthropic API) returns JSON per headline: `sentiment` from −1 to +1, `materiality` 0–3, `event_type` from the event table, and a one-line `why`. Fallback with no API key: FinBERT or a keyword lexicon.
3. **Daily news score** = materiality-weighted mean sentiment, plus headline count z-score (an unusual number of headlines is a signal on its own).
4. **Event flags** — match `event_type` against `event_base_rates.json` and attach the prior.

### Tier 2: the analyst panel (only when materiality = 3)

Triggered by a new major event (CEO change, merger, probe, guidance shock). It runs the actor and scenario method from the original framework, with the LLM doing the work.

1. **Actor map** — one call lists the key actors and fills, for each: what they want, what they can do (votes, vetoes, authority), how credible their statements are vs. past actions, and what limits them.
2. **Scenario tree** — a second call lists 3–5 mutually exclusive outcomes with a time frame, and breaks multi-step paths into stages (announce → approve → close) so stage probabilities multiply.
3. **Panel of personas** — the same evidence goes to 5 personas in separate calls: bull, bear, lawyer/regulator, the board's view, and a base-rate statistician who only sees the event table. Each returns a probability per scenario and its reasoning.
4. **Aggregate** — combine the persona probabilities in log-odds space (geometric mean of odds), then shrink toward the event prior from the table. Report the spread across personas as a disagreement score: high spread = low confidence.
5. **Price per scenario** — the panel estimates the stock price under each outcome; expected value = Σ p × price, compared with the pre-market price.
6. **Tripwires** — the panel lists 3–5 future signals that would move the probabilities (a filing, a vote date, a named person's statement). The next morning's run checks the news for each.

The panel's output is saved to `data/events/<event_id>.json` and re-run when a tripwire fires, so each event has a dated history of probability updates.

### Prompt rules

- Every claim in the panel's output must cite a headline or filing from that day's input; uncited claims are dropped.
- Personas never see each other's answers before answering (independence matters for the aggregate).
- Prompts and model versions are stored with each run, so results can be reproduced and compared.

## Bayesian combiner

The final P(up) starts at the base rate and moves only as far as each signal has earned through its measured track record. Signals are combined in log-odds space, which is the Bayesian way to add independent evidence.

```latex
\text{logit}(p_{\text{final}}) = \text{logit}(p_0) + \sum_i w_i \left[ \text{logit}(p_i) - \text{logit}(p_0) \right]
```

- p₀ = the prior: the stock's unconditional up-rate (about 52–53% for most large stocks), or the gap base rate on a gap day.
- pᵢ = each signal's probability: the ML ensemble, each active conditional base rate, and the news score mapped to a probability.
- wᵢ = each signal's weight, between 0 and 1.

### How weights are set

| Signal | Starting weight | How it is learned |
| --- | --- | --- |
| ML ensemble | min(0.6, 20 × best model BSS) | walk-forward BSS, recomputed weekly |
| Conditional base rates | 0.5 × √(N / 100), capped at 0.5 | grows with sample size N |
| Gap base rate | replaces p₀ when the absolute gap ≥ 1% | matched on similar past gaps |
| News score | 0.35 | refit monthly from the forecast ledger |
| Event panel | 0 until 20 scored events | refit from the ledger |

After 60+ logged forecasts, replace the hand-set weights with a stacking model: a logistic regression whose inputs are each signal's logit and whose target is the actual outcome, trained only on past ledger rows. This learns the weights from data.

### Guards

- Cap the final probability to 35–65% for direction. A daily number outside that range almost always means a bug or overfitting.
- Correlated signals (e.g. RSI-oversold and the ML model, which also sees RSI) are not independent; the cap and the small weights limit double-counting.
- Report an "agreement" line: how many signals point the same way. Agreement with small weights is still a weak signal.

## Report output

One HTML page per morning (`reports/YYYY-MM-DD.html`) with a card per ticker, plus a short Markdown summary that can be pushed to Telegram, email or Slack. Each card follows the same order: range first, direction second, news third, warnings last.

Example card layout (all numbers are made up to show the format):

```markdown
## NVDA  —  pre-market $X.XX  (gap +2.1% vs close $Y.YY)

VOLATILITY   regime HIGH  ·  sigma 3.4%  ·  68% band $A–$B  ·  95% band $C–$D
             band check: 69% of past days closed inside the 68% band

DIRECTION    P(up) 54%   (prior 52% → final 54%)
             ML ensemble 51% (weight 0.10, best BSS 0.005)
             Base rate "extreme up day": 58% up next day, N = 19 (weight 0.22)
             Gap +1.3% to +3.4%: closes above open 47%, gap fills 41%, N = 112
             News score +0.4 (weight 0.35)  ·  agreement 3 of 4 signals up

EVENTS       CEO return (materiality 3)  →  analyst panel ran
             Scenarios: rally holds 20d 55% · partial giveback 30% · full giveback 15%
             Persona spread: high  ·  tripwires: SEC probe update, board statement

TOP NEWS     +0.8  Board reinstates Huang as CEO  (Yahoo, 14h ago)
             -0.3  SEC probe continues  (WSJ, 20h ago)

WARNINGS     direction edge is weak (BSS < 0.01)  ·  news source 2 of 3 failed
```

Rules for the card:

- Every probability shows its N or its weight, so a strong-looking number with no support is visible.
- The "WARNINGS" line is never hidden: stale data, failed sources, no model skill, too few base-rate cases.
- No buy/sell wording. The report shows probabilities and ranges; the decision stays with you.

## Scheduling, ledger and scoring

### When it runs

Schedule in New York time so daylight saving is handled automatically: 8:30 a.m. ET is 21:30 Singapore time in winter (EST) and 20:30 in summer (EDT).

```
CRON_TZ=America/New_York
30 8 * * 1-5   cd ~/premarket && .venv/bin/python -m premarket.run >> logs/run.log 2>&1
```

- Skip market holidays with `exchange_calendars` (calendar `XNYS`); exit cleanly with a one-line "market closed" note.
- On macOS, use `launchd` instead of cron if the laptop sleeps; or run it on a small cloud VM or a GitHub Actions schedule.
- Target run time: under 3 minutes for 3 tickers. Model refits happen weekly (Sunday), not every morning; the morning run only loads saved models and predicts.

### Forecast ledger

Every forecast is appended to `data/forecast_ledger.csv`, one row per ticker per day. This is the memory that makes the system improve.

| Column | Example | Filled when |
| --- | --- | --- |
| date, ticker | 2026-11-24, NVDA | morning run |
| p\_prior, p\_ml, p\_news, p\_baserate, p\_final | 0.52, 0.51, 0.62, 0.58, 0.54 | morning run |
| sigma\_pred, band68\_low, band68\_high | 0.034, … | morning run |
| active\_conditions, event\_types | extreme\_up\_day; ceo\_return | morning run |
| outcome\_up, actual\_ret, inside\_band68 | 1, +0.021, 1 | next morning |
| model\_version, prompt\_version | 2026-11-23, v3 | morning run |

### Scoring (runs at the start of each morning, before the new forecast)

1. Fill yesterday's outcome columns from the latest close.
2. Rolling 60-day Brier score for each signal and for p\_final, against the always-base-rate baseline.
3. Calibration table: of all days forecast at 55–60%, did 55–60% actually go up? Show a reliability chart weekly.
4. Band coverage: share of days inside the 68% band (target 68%; well below means sigma is too low).
5. Weekly summary line in the report: "Last 60 days: p\_final BSS +0.004, ML BSS −0.002, news BSS +0.011, band coverage 66%."

A signal whose rolling BSS stays below zero for 60 days has its weight cut to zero automatically.

## Build plan for Claude Code

Build in five milestones; each one produces a working report, so the system is useful from milestone 1 and gets sharper with each step.

1. **M1 — Data + GARCH range report.** `data.py`, `features.py`, `garch.py`, a plain HTML report with price bands and regime. Done when: the report runs for all 3 tickers and the 68% band coverage check prints.
2. **M2 — Base rates.** `baserates.py` (conditional + gap) and `event_base_rates.json`. Done when: the report lists active conditions with N and the gap table on a gap day.
3. **M3 — ML panel with walk-forward.** `models.py` with the four models, BSS per model, saved models, weekly refit job. Done when: a walk-forward table (accuracy, Brier, BSS per model) prints for each ticker.
4. **M4 — News tier 1 + combiner + ledger.** LLM headline scoring, `combine.py`, `ledger.py`, daily scoring. Done when: p\_final appears with its breakdown and yesterday's row gets its outcome filled.
5. **M5 — Analyst panel + scheduling.** Tier-2 actor/scenario panel for materiality-3 events, cron/launchd setup, Telegram or email push. Done when: it runs unattended for 5 trading days.

**v2 ideas, after 60+ ledger rows:** options implied volatility vs GARCH sigma, Polymarket odds for named events, LightGBM, stacking-model weights, SEC filing alerts.

### Tests to ask for

- No look-ahead: shifting the price series by one day must change every feature on the last row.
- GARCH on simulated GARCH data recovers the true parameters within tolerance.
- Walk-forward never trains on a date at or after its test date.
- Combiner: with all weights 0, p\_final equals p₀; output stays inside the 35–65% cap.
- Offline mode: with network off, the run uses cached data and the report shows the warning.

### Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Overfitting the ML models | fake edge in backtest, losses live | shallow models, walk-forward only, BSS gate, ledger tracking |
| yfinance breaks or rate-limits | no data that morning | disk cache + paid-API fallback slot |
| Over-trusting a confident LLM panel | big bets on a story | panel weight 0 until scored; persona spread shown |
| Event regimes break history | base rates mislead on rare events | show N; treat N < 25 as context only |
| Report read as a trade signal | impulsive trades | no buy/sell wording; warnings line always shown |

### Starter prompt for Claude Code

```markdown
Read DESIGN.md (this doc, exported). Build milestone M1 only:
a Python 3.11 package `premarket/` with data.py, features.py, garch.py, report.py and run.py,
config in config.json (tickers TSLA, NVDA, OPEN). Use yfinance with disk caching,
the `arch` package for GJR-GARCH(1,1) Student-t, and pytest tests for no look-ahead
and GARCH parameter recovery. Stop after M1 and show me the report for today.
```
