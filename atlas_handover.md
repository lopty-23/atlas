# atlas — Project Handover (Phases 1, 2, 4 complete + full results analysis)

Cold-start handover for a new chat session. Carries the full state of the
`atlas` project through the END of Phase 4 and a complete four-stage results
analysis, with docs refreshed and de-AI'd. The next session's first task is a
finer-grained "where does the strategy bleed" diagnostic (§7).

---

## 0. How to work with the user (read first)

- **The user (Isaiah) is a beginner Python programmer** who reads code well but
  writes slowly. He wants to *understand* every piece. Explain code line by line
  when asked, and explain *why* each design decision was made, not just what. He
  pushes back when reasoning is skipped, and halts forward progress until clear
  ("don't proceed until I give the signal") — respect the halt.
- **The assistant's tools run in a SEPARATE SANDBOX** from the user's Mac. The
  assistant CANNOT run project commands or read project files directly. Workflow:
  the assistant composes code/commands, the **user runs them locally and pastes
  output**. To see a file, ask the user to paste/upload it.
- **The user's machine:** macOS, home dir `/Users/lopty/`, project at
  `~/Documents/quant/atlas`.
- **Per-module workflow:** conceptual design (settle forks BEFORE code) -> code ->
  user runs diagnostic on real data -> explain -> tests -> user runs tests ->
  commit (explicit `git add <files>`, never `-A`; messages prefixed `Phase N:` or
  a clear scope) -> push. One logical unit at a time. Verify on real data BEFORE
  writing tests.
- **Conceptual-first is a firm rule.** The user wants design forks settled and
  reasoning laid out before any code is written. Be skeptical of suspiciously
  strong results; flag your own mistakes explicitly rather than shipping them
  (this happened several times in Phase 4 and was always the right move).
- **Docstrings/comments: LEAN.** Module headers a few lines; function docstrings
  short; inline comments only where they prevent a bug. Detailed reasoning lives
  in DECISIONS.md and in chat (chat explanations stay detailed — never trim
  those). The user did a docstring-trimming pass at the checkpoint.
- **Rigor over speed.** The user would rather find a real bug or an honest
  negative result than ship something that looks good. He values being steered
  away from overfitting/look-ahead traps even when an idea is appealing.

---

## 1. What atlas is

A point-in-time-correct research framework for systematic **macro** cross-asset
strategies on free daily data (Yahoo ETFs, FRED/ALFRED macro). Covers the full
research pipeline: revision-aware ingest -> signals -> IC evaluation -> blending
-> sizing -> risk limits -> costed backtest -> performance -> results analysis.
Guiding principle: **methodological correctness over backtest aesthetics**.
Inspired by Stefan Jansen's ML-for-Trading framework concept and AQR Economic
Trend methodology, built as clean tested modules, not notebooks.

---

## 2. Environment & conventions (CRITICAL — do not deviate)

- **`uv`** (v0.11.x): ALWAYS `uv run python ...` and `uv run python -m pytest
  tests/ -v`. NEVER bare `python` (system 3.13 lacks packages; venv is 3.12.3).
- `atlas` package under `src/atlas/`, hatchling, editable via `uv sync`. Editable
  installs pick up NEW modules automatically (no re-sync needed).
- Floats: `np.isclose` / `np.allclose`, never `==`. Type-hint every function.
- **Signal/frame contract: float64 with `np.nan` for missing — NEVER `pd.NA`**
  (see gotcha §8). NaN = abstain/no-position, never coerced to 0 except at
  documented boundaries (engine trade time).
- Tests: hand-crafted known-answer inputs; test the INVARIANT, not a proxy;
  point-in-time perturbation tests (perturb the future, assert the past is
  unchanged). **154 tests, all passing.**
- Data (`data/*.parquet`) gitignored; regenerate via `scripts/run_ingest.py`.
  `.env` holds the FRED key (gitignored). Repo: private `github.com/lopty-23/atlas`.
- **Data caches were last refreshed through 2026-06-17.** A backup of the prior
  caches sits in `data_backup_2026-06/` (the user may have deleted it).

---

## 3. Current state — Phases 1, 2, and 4 COMPLETE; results analysis COMPLETE

(Phase 3 = ML nowcasting, deliberately not started. Phase 4 absorbed the
backtest originally pencilled as Phase 5.)

```
src/atlas/
  data/      ingest.py, point_in_time.py, returns.py, validate.py        (Phase 1)
  signals/   base.py, momentum.py, carry.py, macro_trend.py              (Phase 2)
  evaluation/ ic.py, correlation.py                                       (Phase 2)
  portfolio/ roster.py, blend.py, sizing.py, risk.py                      (Phase 4)
  backtest/  engine.py, performance.py                                    (Phase 4)
  live/      empty (future)
  models/    empty (Phase 3 nowcasting)
scripts/     run_ingest, evaluate_signals, evaluate_correlations, tune_signals,
             build_composite, build_sizing, build_risk, run_backtest,
             evaluate_backtest, evaluate_regimes, evaluate_attribution,
             evaluate_sensitivity
config/universe.yaml   21 ETFs, 23 FRED series, risk_buckets, risk_limits
DECISIONS.md, README.md, atlas_handover.md
tests/       154 tests, all passing
```

### Phase 1–2 in brief (full detail in README / DECISIONS)

- 21 ETFs / 23 FRED series from 2003; 5 parquet caches. `build_pit_macro()`
  builds the daily point-in-time macro frame (fixed lags + ALFRED vintages). It
  is NOT a script — it is a library function each downstream script calls at
  runtime (the lag/vintage correction is recomputed live, never cached, so a fix
  propagates everywhere with no stale intermediate).
- Signals subclass `Signal`: `_compute_raw` -> winsorize (robust median/MAD) +
  normalization. `cross_sectional` = per-date z-score across assets (momentum,
  carry); `time_series` = pass through unchanged, for the macro signals
  normalized over their own history via an EXPANDING z-score (the critical
  no-look-ahead device; also why macro signals reach ±4.5 in 2022).
- **Blend roster (4): TSMomentum, BondCarry, FXCarry, InflationTrend.**
  GrowthTrend DROPPED (COVID publication-lag whipsaw; code retained for Phase 3
  nowcasting reintroduction). BondCarry is the strongest signal (IC 0.039 @1d,
  t=4.9); InflationTrend is regime-conditional (~0 full-sample IC, real edge only
  when inflation is live; IC negatively correlated with BondCarry = ballast).
  BondCarry needs `dividends` + `prices_unadjusted` injected via constructor
  (dist yield = trailing dividends / UNADJUSTED price; warm-up masked 365d).

### Phase 4 — what was built, and the key decisions

**`portfolio/roster.py`** — single source of truth for the 4-signal construction
(BondCarry injection lives ONCE). `ROSTER_SCALES = {"InflationTrend": 2.0}`: an
A-PRIORI "2 sigma is strong" units divisor (NOT fitted), applied so the
time-series-normalized inflation signal (±4.5) doesn't swamp the cross-sectional
ones (±2). Dividing by a constant preserves SHAPE (regime info) — this rescale IS
the regime-conditional weighting: the inflation signal's voice naturally shrinks
toward zero in calm periods and rises in a real regime, with no threshold to fit.

**`portfolio/blend.py` — `blend_signals(signals, weights=None, scales=None)`.**
Per-date, per-asset weighted average over the signals PRESENT (non-NaN) at each
cell, renormalized over the present set (absent signal adds 0 to BOTH numerator
and denominator — the .sum()-drops-NaN fix). Union grid over dates/assets (never
intersection). `scales` divides each signal by a constant first. Denominator
poisoned (`.where(>0)`) before dividing. Does NOT apply the execution lag (that's
the engine's job — double-lagging is the bug to avoid).

**`portfolio/sizing.py` — `compute_target_weights(composite, prices, mode,
target_vol=0.10, vol_window=126)`.** Three stages. (A) mode fork: `long_short` =
composite straight through, **NO demean** (net exposure retained; the only
net-exposure source is InflationTrend's betas summing to ~-4, so the net tilt IS
the inflation view — net short risk/duration when inflation rises); `long_only` =
clip shorts to 0. (B) inverse-vol, w ∝ dir/σ (126d trailing). (C) per-date scalar
k_t = target / sqrt(wᵀΣw·252), Σ = trailing 126d rolling covariance on CURRENT
weights (point-in-time), vectorized via the rolling-cov panel + two contractions.
Output gross-UNCONSTRAINED. Explicit covariance chosen over matrix-free (noise in
Σ matters when OPTIMIZING against it, not when MEASURING a scalar).

**`portfolio/risk.py` — `apply_risk_limits(weights, risk_buckets,
max_position=0.20, max_bucket=0.50, max_gross=3.0)`.** Caps are fractions of
TARGET gross -> fixed ABSOLUTE thresholds (0.60 / 1.50 / 3.0) — the "B2" choice
(an earlier fraction-of-REALIZED-gross version was self-referential and infeasible
on concentrated books; caught by the build_risk.py diagnostic). Proportional
scaling preserves within-group ratios; order per-asset -> bucket -> leverage
(leverage last is exact under absolute caps). Coverage guard raises on
unbucketed/double-bucketed assets. Drawdown throttle deliberately NOT here
(path-dependent -> engine).

**`backtest/engine.py` — `run_backtest(weights, prices, rf, rebalance="ME",
cost_per_side=0.0001)`.** THE place the execution lag lives (nowhere else):
statement order in the daily loop makes targets read at rebalance t earn returns
from t+1. Drift between monthly rebalances (w <- w(1+r)/(1+r_p)); turnover
measured against the DRIFTED book. Costs 1bp/side on one-way traded notional.
Cash/financing leg: r_p = sum(w_i r_i) + (1-net)·rf, rf = DGS3MO/100/252 (letter
O, not zero). Plain Python loop (drift is sequential; clarity over vectorization).
`rebalance_dates()` groups the trading calendar by period and takes each period's
last actual trading day (frequency-agnostic via `freq[0]`).
**FLAGGED simplification: same-rate financing, no borrow fees — flatters
long_short.** No drawdown throttle in v1 (deferred overlay).

**`backtest/performance.py`** — pure functions of a daily return series: cagr,
annualized_vol, sharpe & sortino (**EXCESS over rf**; sortino denominator =
full-sample RMS of negative excess), max_drawdown (equity curve), calmar
(NaN-guarded), time_underwater, hit_rate, `summarize()`. Contract: caller passes
the LIVE slice (engine returns include flat pre-live zeros). No benchmark logic in
the module — comparisons are a script's job.

---

## 4. Results (refreshed, data through 2026-06-17; 1bp/side costs; 2004-02 -> 2026-06)

Live window 5630 days, ~22.3y. The four-stage analysis (beta decomposition,
regime slices, attribution, sensitivity) is COMPLETE and documented in DECISIONS.

**Headline / beta decomposition** (`evaluate_backtest.py`):

|            | Sharpe | alpha/y      | beta | R2   | MaxDD  |
|------------|--------|--------------|------|------|--------|
| long_short | 0.43   | +3.55% (t1.65)| 0.10 | 0.03 | -21.5% |
| long_only  | 0.55   | +2.81% (t1.41)| 0.35 | 0.32 | -32.8% |
| 60/40      | 0.62   | +0.93% (t1.71)| 0.55 | 0.94 | -32.6% |

Other headline numbers: long_short CAGR 5.91%, vol 0.106, Sortino 0.59, Calmar
0.27, TUW 860d, hit 53.4%, turnover 10.12x/y. long_only CAGR 7.78%, vol 0.116.

**The verdict (settled and stress-tested):** long_short is a near-market-neutral
**crisis-alpha diversifier** whose value is as an additive sleeve, NOT a standalone
Sharpe play. The raw-Sharpe ranking INVERTS under decomposition — long_only's
higher Sharpe is mostly beta (R2 0.32); long_short (beta 0.10) is near-pure alpha
and leads on alpha. Neither alpha clears t=2 (long_short t=1.65 is suggestive,
not proven; needs ~1.5x more sample to certify — the normal fate of a true
~0.4-Sharpe edge).

**Regime slices** (`evaluate_regimes.py`, windows fixed on PUBLIC events before
seeing results — do NOT re-fit, do NOT add a "2024-26" regime):

| Regime (window)          | long_short      | long_only        |
|--------------------------|-----------------|------------------|
| GFC (07-10..09-03)       | +16.3% / 1.14   | -3.3% / -0.30    |
| QE calm (12-01..19-12)   | +6.0% / 0.56    | +7.2% / 0.64     |
| COVID (20-02..20-04)     | +33.8% / 1.59   | -45.3% / -1.18   |
| Inflation (21-04..23-07) | +15.2% / 1.12   | +7.0% / 0.49     |

long_short positive in all three crises. Worst DD -21.5%: peak 2018-10-03 ->
trough 2021-02-25 -> recovery 2022-03-07 (directionless 2019 chop + slow
post-COVID recovery; resolves when the inflation regime starts working). NOT a
crisis drawdown — the weakness is trendless calm.

**Diversifier test:** corr(long_short, 60/40) = +0.17; a 70/30 blend (70% 60/40 +
30% long_short) scores Sharpe 0.69 (beats 60/40's 0.62) with MaxDD -19.4% (vs
-32.6%). Higher Sharpe AND shallower drawdown — the portable-alpha case.

**Attribution** (`evaluate_attribution.py`), per-signal leave-one-in Sharpe per
regime (does NOT sum to combined):

| | GFC | QE | COVID | Infl | full |
|---|-----|-----|-------|------|------|
| TSMomentum     | +1.12 | +0.60 | +1.66 | +0.82 | +0.31 |
| BondCarry      | -1.19 | +0.33 | -1.68 | +0.45 | +0.16 |
| FXCarry        | -0.87 | -0.13 | +1.25 | +0.88 | +0.17 |
| InflationTrend | -0.74 | +0.29 | +0.55 | +1.29 | +0.28 |
| COMBINED       | +1.14 | +0.56 | +1.59 | +1.12 | +0.43 |

Momentum drives deflationary crashes (GFC/COVID); InflationTrend owns the
inflation regime (+1.29); BondCarry is momentum's mirror (worst in crashes, a
risk-on harvester); the COMBINED Sharpe (0.43) EXCEEDS every individual signal —
the diversification free lunch. Per-bucket (ex cash leg): credit +3.02% > equity
+2.25% > fx +1.19% > real_assets +0.26% > rates -0.61% > commodities -0.81%;
TOTAL +5.31%/y (vs CAGR 5.91%; gap is the cash leg). Turnover: the leverage cap
REDUCES turnover (-1.94x/y) by clamping high-gross days.

**Sensitivity** (`evaluate_sensitivity.py`, one-at-a-time, no "best" cell picked):
ROBUST on 3 of 4 — vol window {63/126/252} -> Sharpe 0.42/0.43/0.43; leverage cap
{2/3/4} -> 0.40/0.43/0.41 (maxDD drift is the cap working); cost {0.5/1/2/5bps} ->
alpha +3.60/+3.55/+3.44/+3.14% (survives 5bp). The ONE dependency: rebalance
frequency is a THRESHOLD — W 0.41 / ME 0.43 / **QE 0.10** (quarterly breaks it;
the risk machinery can't react to vol spikes if the book drifts a quarter
untouched). Monthly is the lowest-cost viable cadence, not arbitrary.

**STANDING CAVEATS (do not drop, stated in docs):** (1) n=1 inflation regime.
(2) Same-rate financing / no borrow fees flatters long_short specifically.
(3) Every regime slice is low-N — behavioral illustrations, not significant
sub-period claims. (4) Neither full-sample alpha clears t=2. (5) Sensitivity is
SAME-SAMPLE robustness, NOT out-of-sample (walk-forward is the gap, see §7).

NOTE on the refresh: structural numbers (betas, R2, regime Sharpes, drawdown
dates, the 1.65 t-stat, the sensitivity pattern) were byte-identical before and
after the data refresh — the changes were pure third-decimal sample-noise and the
verdict was unchanged. Do NOT re-tune to recover any prior numbers (overfitting).

---

## 5. Documentation state

- **README.md** — portfolio-facing, rewritten and de-AI'd at the checkpoint;
  leads with methodology, frames every number with its caveat, fresh numbers
  baked in. 8 sections.
- **DECISIONS.md** — engineering log, newest-first, de-AI'd prose but bold
  scannable labels retained (it's a log; structure aids findability). 9 sections,
  fresh numbers baked in. Contains all Phase 1/2/4 design decisions and all four
  results-analysis writeups.
- **atlas_math_reference.md** — a standalone math/stats reference the user keeps
  outside the repo (pitched for "comfortable with basic stats"): every quantity
  with what-it-is / formula / how-atlas-computes / how-to-interpret, plus
  interpretation thresholds and a cheat-sheet. NOT in the repo.
- A de-AI editing skill the user is building: hunt em-dash overuse, "not X but Y"
  seesaws, relentless tricolons, hedge/signpost words, bold-sentence paragraph
  openers, CAPS emphasis, over-qualification. README got the full flattening;
  DECISIONS kept bold labels as navigation. The user owns the final voice — coach,
  don't ghost-write.

---

## 6. Hard-won gotchas (avoid repeating these)

- **Vintage value-selection**: per vintage take the LATEST observation_date (sort
  by tiebreaker before first/last). A dedup bug here once injected 1940s values
  onto 2003 dates and corrupted all vintage macro signals. Fixed + regression-
  tested in test_no_lookahead.py.
- **Publication lag can INVERT a macro signal in fast shocks** (killed GrowthTrend
  in COVID). Nowcasting is the fix.
- **Full-sample IC masks regime-conditional signals** (saved InflationTrend,
  condemned GrowthTrend). Always check regime-conditional IC.
- **Per-date weight renormalization** for any weighted combo of components with
  different start dates — divide by PRESENT weight, never a fixed total.
- **Look-ahead acid test**: feed a signal future info; its IC must DEGRADE. (The
  ML version of this is feature leakage — critical for Phase 3.)
- **Test the invariant, not a proxy** (the bound test caught the NaN-weighting bug;
  a magnitude proxy would not have).
- **Distribution yield denominator = UNADJUSTED price.**
- **`pd.NA` poisons dtype**: one `replace(0.0, pd.NA)` made InflationTrend
  object-dtype and crashed the blend's division (object 0/0 raises;
  numpy float64 0/0 = nan). Contract: float64 + np.nan everywhere.
- **Poison denominators BEFORE dividing**: `denom.where(denom > 0)` then divide
  (x/NaN = NaN, never raises). Used in blend, sizing, calmar.
- **Fraction-of-REALIZED-gross caps are self-referential / infeasible** on
  concentrated books. Cap against a STABLE denominator (target gross -> absolute
  thresholds). The real-data diagnostic caught this where unit tests would not
  have — always run the diagnostic BEFORE writing tests.
- **Backtest lag lives in exactly ONE place** (engine statement order).
  Blend/sizing/risk never shift; double-lagging is the bug to avoid.
- **Turnover measured against the DRIFTED book**, not stale targets
  (constant-weights-between-rebalances is fictional and understates costs).
- **Raw Sharpe comparisons across books with different net exposure are
  beta-confounded** — decompose (alpha/beta) before concluding.

---

## 7. NEXT: forward agenda (in priority order)

**1. "Where does the strategy bleed" diagnostic — THE IMMEDIATE NEXT TASK.**
Conceptual-first, as always. Goal: a finer-grained characterization of the
strategy's weak periods than the four coarse regime windows. Likely shape (to be
designed with the user): roll Sharpe and/or drawdown over a trailing window across
the whole 2004-2026 sample, identify the worst sub-periods MECHANICALLY (not only
inside hand-picked windows), and test whether the QE-calm weakness is uniform or
concentrated. The standing conclusion it refines: the strategy's weakness is
TRENDLESS, NON-INFLATIONARY CALM (carry grinding alone, momentum and inflation
quiet) — the structural complement of its crisis strength.
**CRITICAL FRAMING: this is diagnostic-only. Characterize the weakness; do NOT
build a regime filter to "fix" it — fitting a regime-detector to the specific
historical calm periods that hurt is the curve-fit trap.** A new analysis script
(`evaluate_*`), no library module, no test file (correctness is structural, like
the other results scripts).

**2. Financing-spread refinement.** Replace the same-rate-financing simplification
(the flagged item that flatters long_short) with a financing spread + short-borrow
fees. Engine change + DECISIONS update. Bounded and well-scoped; directly retires a
standing caveat. Re-run results afterward (expect long_short alpha to shave down
somewhat — that's the honest direction).

**3. Walk-forward / out-of-sample validation.** The one analytical gap the
results phase could not close (everything so far is in-sample). Re-fit/re-decide
on a rolling past window, test on the next unseen chunk, concatenate the
out-of-sample pieces. A STRONGER test of whether the edge is real than
accumulating sample or sweeping knobs; directly addresses the biggest standing
caveat. Medium difficulty.

**4. Phase 3 — ML nowcasting layer (`models/`).** The one genuinely valuable ML
use here: predict macro releases (inflation, growth) from higher-frequency data
BEFORE official publication, to beat the publication lag. This is also
**GrowthTrend's path back** (it was dropped only because the lag whipsawed it; a
nowcast that front-runs the release is the fix), AND the principled way to make
the inflation signal more regime-aware. Highest difficulty; introduces an ML
testing surface where feature-leakage is the look-ahead analogue to guard against.
Direct ML return-prediction is NOT expected to beat simple signals (learning
baseline only).

**5. `live/` scaffolding.** Wire the existing stack to generate today's target
weights from current data (paper-trading-ready). Lower difficulty; makes atlas
operable; natural portfolio-completeness story.

**Other deferred items:** drawdown-throttle overlay (its own design pass, free
parameters); commodity roll-yield carry (needs futures-curve data, not free);
graded macro betas (Phase 3 revisit); IC-weighted blend weights (blend is already
parameterized; 1/N default kept); docstring-condensing already largely done.

---

## 8. Suggested first message to the new session

Upload this file and say: "This is the handover for my atlas quant project,
Phases 1, 2, 4 complete with a full results analysis (verdict in §4, refreshed
numbers through 2026-06-17). I want to start the next task: the 'where does the
strategy bleed' diagnostic per §7 — a finer-grained look at the weak periods,
diagnostic-only, NOT a regime filter. Conceptual design first, as always." The
assistant should confirm the separate-sandbox workflow and the
explain-everything / conceptual-first expectations before starting, and read the
relevant files (it cannot run them).
