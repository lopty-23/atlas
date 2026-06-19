# atlas

A point-in-time-correct research framework for systematic **macro** cross-asset
strategies on free daily data. It covers the full pipeline: ingesting
revision-aware macro data, building signals, testing their predictive power,
blending them into a portfolio, and backtesting with realistic costs.

Extra attention is placed to ensure no look-ahead bias and include realistic execution
Results will also be reported with their caveats (no p-hacking here as can be seen from
how the numbers aren't that good in the first place).

> Status: Phases 1, 2, and 4 complete (data layer, signals, evaluation,
> portfolio construction, backtest, performance, results analysis). 154 tests
> passing. Phase 3 (ML nowcasting) and live deployment are scoped but not built.

---

## What it does

The pipeline runs from raw data to a scored equity curve as tested modules, not
notebooks:

**1. Point-in-time data layer.** Ingests ETF prices and dividends (Yahoo) and
~23 macro series (FRED), and reconstructs what was knowable on each historical
date: real publication lags for market series, ALFRED vintage data for
revision-heavy releases like CPI and payrolls. 

**2. Signals.** An abstract `Signal` base class standardises the roster:
vol-scaled time-series momentum, bond carry from distribution yield, FX carry
from rate differentials, and an inflation-trend macro signal. Each is
point-in-time safe and tested against future-data deviation.

**3. Evaluation.** A signal-agnostic Information Coefficient (IC) toolkit:
cross-sectional IC and decay curves, ICIR with Newey-West (HAC) standard errors
for overlapping-horizon autocorrelation, quintile spreads, and inter-signal
correlation. 

**4. Portfolio construction.** Blends the signals (per-date renormalisation over
whichever signals are present, with a shape-preserving rescale so the
time-series-normalised macro signal doesn't swamp the cross-sectional ones),
sizes positions by inverse volatility scaled to a portfolio vol target (10%), and
applies risk limits (per-asset, per-class, gross-leverage caps). Runs for both 
long-only and long-short.

**5. Backtest and performance.** Simulates the book with a 1-day execution lag,
monthly rebalancing, position drift between rebalances, transaction costs, and a
risk-free cash leg, then scores it (CAGR, Sharpe and Sortino on excess returns,
Calmar, max drawdown, time underwater).

---

## Signals and the keep/drop discipline

Signals were evaluated by IC (with a 1-day execution lag and HAC-corrected
significance), then filtered by regime and inter-signal correlation. Four of
five candidates made the blend.

| Signal | Standalone edge | Verdict |
|---|---|---|
| Bond carry | Strong, highly significant (t > 5 across horizons) | Core signal |
| FX carry | Significant to ~63d (t > 2); thin 4-asset cross-section | Kept |
| Momentum | Modest, slow; independent of carry | Kept (diversifier) |
| Inflation trend | ~0 full-sample, but a real edge when inflation is live | Kept (conditional) |
| Growth trend | Whipsawed by publication lag in fast shocks (COVID) | Dropped, pending nowcasting |

The lesson that drove these calls: full-sample IC hides regime-conditional
signals. Inflation trend looks dead on average but has a real edge when inflation
is live, so it was kept. Growth trend looks viable on average but has a
structural failure in fast shocks: its data arrives inside the publication lag,
so it went maximally defensive just as markets bottomed in COVID. Hence, it was
dropped, with its code retained for reintroduction once a nowcasting layer can
beat the lag. The evidence for each call is in [`DECISIONS.md`](DECISIONS.md).

---

## Results

The strategy was analysed in four stages: a market-beta decomposition, regime
slices, per-signal attribution, and sensitivity sweeps. Headline numbers over
2004–2026, net of 1bp/side costs:

| | Long-short | Long-only | 60/40 benchmark |
|---|---|---|---|
| Sharpe (excess) | 0.43 | 0.55 | 0.62 |
| Annualised alpha vs SPY | +3.6% (t = 1.65) | +2.8% (t = 1.41) | +0.9% (t = 1.71) |
| Market beta | 0.10 | 0.35 | 0.55 |
| Max drawdown | −21.5% | −32.8% | −32.6% |

Raw Sharpe is misleading here given that it does not account for beta. 
The long-short book is near market-neutral (beta 0.10), so almost all of 
its return is alpha, which is why it leads on alpha. Neither alpha is 
statistically decisive at this sample length (both t < 2), more data is needed to 
verify.

The long-short book is a crisis-alpha diversifier. Sliced by regime, it made
strong, high-Sharpe returns in all three stress periods — the GFC, COVID, and the
2021–23 inflation — while a conventional 60/40 portfolio suffered. Its best
regimes are the market's worst. Attribution confirms why this happens: momentum
drives the deflationary-crash protection (GFC, COVID), inflation trend drives the
inflation-regime protection (2021–23), and carry harvests the calm periods. Because
the signals' good and bad regimes don't coinced, that's why the combined book has a 
higher full-sample Sharpe than any individuals signal.

The long-short book is only ~0.17 correlated with 60/40, so adding a 30% sleeve raises 
the blended Sharpe to 0.69 (from 0.62) and cuts max drawdown from −33% to −19%. This is
perhaps the most useful contribution given that it cannot beat the 60/40 standalone.

The conclusions hold up under parameter changes (volatility window, leverage cap, and 
transaction cost leave the verdict unchanged), except rebalancing frequency. However,
this is not because overfitting occured, but because quarterly rebalancing breaks the 
strategy by being too slow to respond to signal changes. 

Caveats (also in [`DECISIONS.md`](DECISIONS.md)): there's only one major
inflation regime in the sample, so that result is n = 1. The backtest borrows and
lends at the same rate with no short-borrow fees, which flatters the long-short
book specifically. The sensitivity sweeps test parameter fragility on the same
22-year history; they are not out-of-sample validation, which is still future
work. And neither alpha clears the conventional t > 2 bar.

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

    # 4. Run the full strategy backtest (both modes, with costs)
    uv run python scripts/run_backtest.py

    # 5. Results analysis
    uv run python scripts/evaluate_backtest.py     # beta decomposition vs SPY
    uv run python scripts/evaluate_regimes.py      # regime slices + diversifier test
    uv run python scripts/evaluate_attribution.py  # per-signal / per-bucket attribution
    uv run python scripts/evaluate_sensitivity.py  # parameter sensitivity sweeps

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
    ├── portfolio/     # portfolio construction
    │   ├── roster.py          # single source of truth for the signal roster
    │   ├── blend.py           # signal blending (renormalised, scale-reconciled)
    │   ├── sizing.py          # inverse-vol + portfolio vol-targeting; LS/LO modes
    │   └── risk.py            # per-asset / per-class / gross-leverage caps
    ├── backtest/      # historical simulation
    │   ├── engine.py          # lag, drift, monthly rebalance, costs, cash leg
    │   └── performance.py     # CAGR, Sharpe/Sortino, Calmar, drawdown, etc.
    ├── live/          # (upcoming) live signal generation
    └── models/        # (upcoming) ML nowcasting layer

    scripts/               # runnable entry points (ingest, evaluation, backtest, results)
    tests/                 # 154 tests
    config/universe.yaml   # asset universe, FRED series, risk buckets + limits
    DECISIONS.md           # design decisions with supporting evidence

---

## Methodology notes

- **No look-ahead.** Macro releases are lagged by their real publication delay,
  and vintage series use only the data published by each date. Signals are tested
  against future-data perturbation, IC uses a 1-day execution lag, and the
  backtest applies that lag in one place (weights today depend on data through
  yesterday).
- **Overlap-aware significance.** Multi-horizon forward returns overlap day to
  day, which inflates naive significance. ICIR, IC t-stats, and alpha t-stats use
  Newey-West (HAC) standard errors so slow signals aren't overstated.
- **Carry from distribution yield.** Bond carry uses each ETF's trailing
  distribution yield (dividends / *unadjusted* price) minus the short rate,
  rather than a credit-spread proxy — one consistent, free, point-in-time source.
- **Inverse-vol sizing with point-in-time vol-targeting.** Positions are scaled
  by inverse trailing volatility so each asset contributes comparable risk, then
  the book is scaled to a ~10% annualised vol target using only trailing data
  (current weights on past returns), never full-sample volatility.
- **Risk limits as backstops.** Per-asset, per-class, and gross-leverage caps are
  loose enough to let the signal express and bind only to stop any single name,
  class, or the overall leverage from dominating. They are not tuned to flatter
  the backtest.

---

## Scope and disclaimer

This is a research and learning project, and a portfolio piece. The strategy is
not deployed, the financing model is simplified (and flatters the long-short
book), the inflation-regime result is n = 1, and the robustness analysis is
in-sample rather than walk-forward. Nothing here is investment advice; backtested
or IC-derived edges are not guarantees of live performance.
