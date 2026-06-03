# atlas

A point-in-time-correct research framework for systematic **macro** cross-asset
strategies, built on free daily data. Atlas covers the research half of the
quant pipeline end to end: ingesting revision-aware macro data, generating
signals, and rigorously evaluating their predictive power before any of them is
trusted in a portfolio.

The emphasis is on **methodological correctness over backtest aesthetics** — no
look-ahead bias, realistic execution assumptions, and signals that have to earn
their place on evidence rather than on a pretty equity curve.

> Status: Phases 1–2 complete (data layer + signals + evaluation). Portfolio
> construction and backtesting (Phases 4–5) are scaffolded but not yet
> implemented. 84 tests passing.

---

## What it does

1. **Point-in-time data layer.** Ingests ETF prices/dividends (Yahoo) and ~23
   macro series (FRED), and reconstructs *what was actually knowable on each
   historical date* — applying real publication lags to market series and using
   ALFRED vintage data for revision-heavy releases (CPI, GDP, payrolls). This
   is the foundation that keeps every downstream signal honest.

2. **Signals.** A small abstract `Signal` base class standardises five
   implemented signals (vol-scaled time-series momentum, bond carry from
   distribution yield, FX carry from rate differentials, and growth- and
   inflation-trend macro signals). Each signal is point-in-time safe and tested
   against future-data perturbation.

3. **Evaluation.** A signal-agnostic Information Coefficient (IC) toolkit:
   cross-sectional IC and decay curves, ICIR with Newey-West (HAC) standard
   errors to handle overlapping-horizon autocorrelation, quintile-return
   spreads, and inter-signal correlation analysis. This layer decides which
   signals are real.

---

## Key findings

Signals were evaluated by IC (with a 1-day execution lag and HAC-corrected
significance), then filtered by regime-conditional analysis and inter-signal
correlation. Four of five signals were selected for the eventual blend.

| Signal | Standalone edge | Verdict |
|---|---|---|
| **Bond carry** | Strong, highly significant (t > 5 across horizons) | Core signal |
| **FX carry** | Significant to ~63d (t > 2); thin 4-asset cross-section | Kept |
| **Momentum** | Modest, slow; independent of carry | Kept (diversifier) |
| **Inflation trend** | ~0 full-sample, but +0.08 IC in high-inflation regimes | Kept (conditional) |
| **Growth trend** | Whipsawed by publication lag in fast shocks (COVID) | **Dropped**, pending nowcasting |

The keep/drop decisions are documented with their evidence in
[`DECISIONS.md`](DECISIONS.md). The central lesson: full-sample IC masks
regime-conditional signals — inflation trend looks dead on average but has a
real edge when inflation is live, while growth trend looks viable on average but
has a structural failure mode in fast shocks.

---

## Setup

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

    git clone https://github.com/lopty-23/atlas.git
    cd atlas
    uv sync

Create a `.env` from the template and add a free [FRED API
key](https://fred.stlouisfed.org/docs/api/api_key.html):

    cp .env.example .env
    # edit .env: FRED_API_KEY=your_key_here

---

## Usage

All commands run through uv.

    # 1. Ingest data (writes parquet caches to data/)
    uv run python scripts/run_ingest.py

    # 2. Evaluate each signal's predictive power (IC, decay, quintiles)
    uv run python scripts/evaluate_signals.py

    # 3. Inter-signal correlation analysis
    uv run python scripts/evaluate_correlations.py

    # Run the test suite
    uv run python -m pytest tests/ -v

---

## Structure

    src/atlas/
    ├── data/          # point-in-time data layer
    │   ├── ingest.py          # fetch ETF prices/dividends + FRED (raw & vintage)
    │   ├── point_in_time.py   # publication lags + ALFRED vintage alignment
    │   ├── returns.py         # forward/trailing returns, vol, distribution yield
    │   └── validate.py        # data validation
    ├── signals/       # signal library
    │   ├── base.py            # abstract Signal (winsorize, z-score, normalization)
    │   ├── momentum.py        # vol-scaled time-series momentum
    │   ├── carry.py           # bond carry (distribution yield) + FX carry
    │   └── macro_trend.py     # growth- and inflation-trend macro signals
    ├── evaluation/    # signal evaluation
    │   ├── ic.py              # IC, decay, ICIR (Newey-West), quintiles
    │   └── correlation.py     # positioning + IC co-movement matrices
    ├── portfolio/     # (upcoming) signal blending + position sizing
    ├── backtest/      # (upcoming) historical simulation
    ├── live/          # (upcoming) live signal generation
    └── models/        # (upcoming) ML nowcasting layer

    scripts/               # runnable entry points (ingest, evaluation)
    tests/                 # 84 tests
    config/universe.yaml   # asset universe + FRED series definitions
    DECISIONS.md           # design decisions with supporting evidence

---

## Methodology notes

- **No look-ahead.** Macro releases are lagged by their real publication delay;
  vintage series use only the data actually published by each date. Signals are
  tested against future-data perturbation, and IC is computed with a 1-day
  execution lag (you act the day *after* a signal is observed).
- **Overlap-aware significance.** Multi-horizon forward returns overlap day to
  day, inflating naive significance. ICIR and t-stats use Newey-West (HAC)
  standard errors so the reliability of slow signals isn't overstated.
- **Carry from distribution yield.** Bond carry uses each ETF's actual trailing
  distribution yield (dividends ÷ unadjusted price) minus the short rate, rather
  than a credit-spread proxy — one consistent, free, point-in-time-safe source.

---

## Scope and disclaimer

This is a research and learning project. The strategy is not deployed, the
backtest is not yet built, and nothing here is investment advice. Backtested or
IC-derived edges are not guarantees of live performance.
