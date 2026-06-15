# atlas

A point-in-time-correct research framework for systematic **macro** cross-asset
strategies, built entirely on free daily data. Atlas runs the full research
pipeline end to end: ingesting revision-aware macro data, generating signals,
rigorously evaluating their predictive power, blending them into a portfolio,
and backtesting that portfolio with realistic costs.

The emphasis throughout is **methodological correctness over backtest
aesthetics** — no look-ahead bias, realistic execution assumptions, signals that
have to earn their place on evidence rather than on a pretty equity curve, and
results reported with their caveats rather than their best face. The headline
results below are deliberately modest and honest; the value of the project is in
the process that produced them.

> Status: Phases 1, 2, and 4 complete (data layer, signals, evaluation,
> portfolio construction, backtest, performance, and a full results analysis).
> 154 tests passing. Phase 3 (ML nowcasting) and live deployment are scoped but
> not built.

---

## What it does

The pipeline is a single chain from raw data to a scored equity curve, built as
clean tested modules rather than notebooks:

**1. Point-in-time data layer.** Ingests ETF prices/dividends (Yahoo) and ~23
macro series (FRED), and reconstructs *what was actually knowable on each
historical date* — applying real publication lags to market series and using
ALFRED vintage data for revision-heavy releases (CPI, GDP, payrolls). This is
the foundation that keeps every downstream signal honest, and it is where a
latent look-ahead bug was caught and fixed during development (decades-old
historical values were leaking onto modern dates through a vintage-alignment
flaw; the fix is regression-tested, and a "look-ahead acid test" is now standard).

**2. Signals.** A small abstract `Signal` base class standardises the roster:
vol-scaled time-series momentum, bond carry from distribution yield, FX carry
from rate differentials, and an inflation-trend macro signal. Each signal is
point-in-time safe and tested against future-data perturbation.

**3. Evaluation.** A signal-agnostic Information Coefficient (IC) toolkit:
cross-sectional IC and decay curves, ICIR with Newey-West (HAC) standard errors
to handle overlapping-horizon autocorrelation, quintile-return spreads, and
inter-signal correlation. This layer decides which signals are real *before* any
of them touches a portfolio.

**4. Portfolio construction.** Blends the signals (per-date renormalisation over
whichever signals are present, with a shape-preserving rescale so a
time-series-normalised macro signal does not swamp the cross-sectional ones),
sizes positions by inverse volatility scaled to a portfolio volatility target,
and applies risk limits (per-asset, per-asset-class, and gross-leverage caps).
Supports a long-short and a long-only mode from one code path.

**5. Backtest & performance.** Simulates the book with a 1-day execution lag,
monthly rebalancing, position drift between rebalances, transaction costs, and a
risk-free cash leg, then scores it (CAGR, Sharpe/Sortino on excess returns,
Calmar, max drawdown, time underwater).

---

## Signals and the keep/drop discipline

Signals were evaluated by IC (with a 1-day execution lag and HAC-corrected
significance), then filtered by regime-conditional analysis and inter-signal
correlation. Four of five candidate signals were selected for the blend.

| Signal | Standalone edge | Verdict |
|---|---|---|
| **Bond carry** | Strong, highly significant (t > 5 across horizons) | Core signal |
| **FX carry** | Significant to ~63d (t > 2); thin 4-asset cross-section | Kept |
| **Momentum** | Modest, slow; independent of carry | Kept (diversifier) |
| **Inflation trend** | ~0 full-sample, but a real edge when inflation is live | Kept (conditional) |
| **Growth trend** | Whipsawed by publication lag in fast shocks (COVID) | **Dropped**, pending nowcasting |

The central lesson, and the reason this matters: **full-sample IC masks
regime-conditional signals.** Inflation trend looks dead on average but has a
genuine edge when inflation is live; growth trend looks viable on average but
has a structural failure mode in fast shocks (its bad data arrives inside the
publication lag, so it went maximally defensive exactly as markets bottomed in
COVID). Inflation trend was *kept on conditional evidence*; growth trend was
*dropped despite a positive-looking average*, with its code retained for
reintroduction once a nowcasting layer can beat the publication lag. Each
decision is documented with its evidence in [`DECISIONS.md`](DECISIONS.md).

---

## Results (honestly)

The strategy was analysed in four stages: a market-beta decomposition, regime
slices, per-signal attribution, and sensitivity sweeps. Headline numbers over
2004–2026, net of 1bp/side costs:

| | Long-short | Long-only | 60/40 benchmark |
|---|---|---|---|
| Sharpe (excess) | 0.43 | 0.56 | 0.62 |
| Annualised alpha vs SPY | **+3.6%** (t = 1.65) | +2.9% (t = 1.44) | +1.0% (t = 1.73) |
| Market beta | 0.10 | 0.35 | 0.55 |
| Max drawdown | −21.5% | −32.8% | −32.6% |

What the analysis actually shows — and what it does *not*:

**The raw-Sharpe ranking is misleading; alpha tells a different story.**
Long-only's higher Sharpe is largely market beta in costume (R² 0.32 vs SPY) —
return anyone can buy with an index fund. The long-short book is near
market-neutral (beta 0.10), so almost all of its return is alpha, and on alpha
it leads. Neither alpha is statistically decisive at this sample length (both
t < 2); the long-short edge is suggestive, not proven, and would need roughly
another decade of data to certify.

**The long-short book is a crisis-alpha diversifier.** Sliced by regime, it made
strong, high-Sharpe returns in all three stress periods — the GFC, COVID, and
the 2021–23 inflation — while a conventional 60/40 portfolio suffered. Its best
regimes are the market's worst. Per-signal attribution confirms the mechanism:
*momentum* drives the deflationary-crash protection (GFC, COVID), *inflation
trend* drives the inflation-regime protection (2021–23), and *carry* harvests
the calm periods. Tellingly, the combined four-signal book has a higher
full-sample Sharpe than any individual signal — the signals' good and bad
regimes don't coincide, so blending them is a genuine diversification gain.

**It is additive to a beta portfolio, not a replacement for one.** Because the
long-short book is only ~0.17 correlated with 60/40, adding a 30% sleeve of it to
a 60/40 portfolio raises the blended Sharpe to 0.70 (from 0.62) *and* cuts the
max drawdown from −33% to −19% — higher return and shallower losses together,
the signature of a real diversifier. This is the honest case for the strategy:
not "it beats 60/40 standalone" (it doesn't on raw Sharpe), but "it improves a
portfolio that holds it."

**The conclusions are robust to parameter choice — with one documented
constraint.** Sensitivity sweeps over the volatility window, leverage cap, and
transaction cost leave the verdict essentially unchanged (the edge survives even
a punitive 5bp/side cost). The one real dependency is rebalancing frequency:
monthly works, but quarterly breaks the strategy (the risk machinery can't react
to volatility spikes if the book drifts untouched for a quarter). Monthly is
therefore an operational requirement, not an arbitrary choice.

**Honest caveats (also in [`DECISIONS.md`](DECISIONS.md)):** there is only one
major inflation regime in the sample, so that result is n = 1. The backtest
assumes borrowing and lending at the same rate with no short-borrow fees, which
flatters the long-short book specifically. The sensitivity analysis tests
fragility to parameter choices on the same 22-year history — it is not
out-of-sample validation, which remains future work. And neither alpha clears
the conventional t > 2 significance bar.

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

- **No look-ahead.** Macro releases are lagged by their real publication delay;
  vintage series use only the data actually published by each date. Signals are
  tested against future-data perturbation, IC is computed with a 1-day execution
  lag, and the backtest applies that lag in exactly one place (weights today are
  a function of data through yesterday).
- **Overlap-aware significance.** Multi-horizon forward returns overlap day to
  day, inflating naive significance; ICIR, t-stats, and the alpha t-stats use
  Newey-West (HAC) standard errors so the reliability of slow signals isn't
  overstated.
- **Carry from distribution yield.** Bond carry uses each ETF's actual trailing
  distribution yield (dividends ÷ *unadjusted* price) minus the short rate,
  rather than a credit-spread proxy — one consistent, free, point-in-time-safe
  source.
- **Inverse-vol sizing with point-in-time vol-targeting.** Positions are scaled
  by inverse trailing volatility so each asset contributes comparable risk, then
  the whole book is scaled to a ~10% annualised volatility target using only
  trailing data (current weights on past returns) — never full-sample volatility.
- **Risk limits as anti-domination backstops.** Per-asset, per-class, and
  gross-leverage caps are set loose enough to let the signal express and bind
  only to prevent any single name, class, or the overall leverage from
  dominating; they are deliberately not tuned to flatter the backtest.

---

## Scope and disclaimer

This is a research and learning project, and a portfolio piece. The strategy is
not deployed, the financing model is simplified (and flatters the long-short
book), the inflation-regime result is n = 1, and the robustness analysis is
in-sample rather than walk-forward. Nothing here is investment advice;
backtested or IC-derived edges are not guarantees of live performance. The point
of atlas is the methodology and the honesty of the analysis, not the headline
numbers.
