# atlas — Project Handover (Phase 4 COMPLETE: portfolio construction + backtest)

Cold-start handover for a new chat session. Carries the full state of the
`atlas` project through the END of Phase 4 (signal blending, position sizing,
risk limits, backtest engine, performance metrics — all built, tested, and run
on real data). The next session's work is the RESULTS conversation: beta
decomposition, regime slices, attribution, sensitivity sweeps (agenda in §7).

---

## 0. How to work with the user (read first)

- **The user is a beginner Python programmer** who reads code well but writes
  slowly. They want to *understand* every piece. Explain code line by line when
  asked, and explain *why* each design decision was made. They push back when
  reasoning is skipped, and will halt forward progress until they're clear —
  respect the halt ("don't proceed until I give the signal").
- **The assistant's tools run in a SEPARATE SANDBOX** from the user's Mac. The
  assistant CANNOT run project commands or read project files. Workflow: the
  assistant composes code/commands, the **user runs them locally and pastes
  output**. To see a file, ask the user to paste/upload it.
- **The user's machine:** macOS, home dir `/Users/lopty/`, project at
  `~/Documents/quant/atlas`.
- **Per-module workflow:** conceptual design (settle forks BEFORE code) → code →
  user runs diagnostic on real data → explain → tests → user runs tests → commit
  (explicit `git add <files>`, never `-A`; messages prefixed `Phase N:`) → push.
  One logical unit at a time. Verify on real data BEFORE writing tests.
- **Docstrings/comments: LEAN.** Module headers a few lines; function docstrings
  one-liners plus params where non-obvious; inline comments only where they
  prevent a bug. Detailed reasoning goes in DECISIONS.md and in chat. (Chat
  explanations stay detailed — never trim those.) blend.py/sizing.py predate
  this bar; a condensing pass is a flagged TODO.
- **The user values rigor over speed.** Be skeptical of strong results; verify
  rather than assume. When the assistant's own code or argument has a flaw, say
  so explicitly rather than shipping it (this happened twice in Phase 4 — the
  dead-code vol-scalar draft and the B1→B2 cap fix — and calling it out was the
  right move both times).

---

## 1. What atlas is

A point-in-time-correct research framework for systematic **macro** cross-asset
strategies on free daily data (Yahoo ETFs, FRED/ALFRED macro). Covers the full
research pipeline: revision-aware ingest → signals → IC evaluation → blending →
sizing → risk limits → costed backtest → performance. Guiding principle:
**methodological correctness over backtest aesthetics**. Inspired by Stefan
Jansen's ML-for-Trading framework concept and AQR Economic Trend methodology,
built as clean tested modules, not notebooks.

---

## 2. Environment & conventions (CRITICAL — do not deviate)

- **`uv`** (v0.11.x): ALWAYS `uv run python ...` / `uv run python -m pytest
  tests/ -v`. NEVER bare `python` (system 3.13 lacks packages; venv is 3.12.3).
- `atlas` package under `src/atlas/`, hatchling, editable via `uv sync`.
  Editable installs pick up NEW modules automatically (no re-sync needed).
- Floats: `np.isclose`/`np.allclose`, never `==`. Type-hint everything.
- **Signal/frame contract: float64 with `np.nan` for missing — NEVER `pd.NA`**
  (see gotcha §8). NaN = abstain/no-position, never coerced to 0 except at
  explicitly documented boundaries (engine trade time).
- Tests: hand-crafted known-answer inputs; test the INVARIANT, not a proxy;
  point-in-time perturbation tests (perturb future, assert past unchanged).
  **154 tests, all passing.**
- Data (`data/*.parquet`) gitignored; regenerate via `scripts/run_ingest.py`.
  `.env` holds FRED key. Repo: private `github.com/lopty-23/atlas`.

---

## 3. Current state — Phases 1, 2, and 4 COMPLETE

(Phase 3 = ML nowcasting layer, deliberately not started; Phase 4 absorbed the
backtest originally pencilled as Phase 5.)

```
src/atlas/
├── data/          # Phase 1: ingest, point_in_time, returns, validate
├── signals/       # Phase 2: base, momentum, carry, macro_trend
├── evaluation/    # Phase 2: ic (IC/decay/ICIR/quintiles), correlation
├── portfolio/     # Phase 4: blend.py, roster.py, sizing.py, risk.py
├── backtest/      # Phase 4: engine.py, performance.py
├── live/          # empty (future)
└── models/        # empty (Phase 3 nowcasting)
scripts/           # run_ingest, evaluate_signals, evaluate_correlations,
                   # tune_signals, build_composite, build_sizing, build_risk,
                   # run_backtest
config/universe.yaml   # 21 ETFs, 23 FRED series, risk_buckets, risk_limits
DECISIONS.md           # design decisions w/ evidence — READ for any Phase 4 work
```

### Phase 1–2 in brief (details in README/DECISIONS)

- 21 ETFs / 23 FRED series from 2003; 5 parquet caches; `build_pit_macro` gives
  a daily PIT macro frame (fixed lags + ALFRED vintages).
- Signals subclass `Signal`: `_compute_raw` → winsorize + normalization
  (`cross_sectional` z-score per date, or `time_series` = unchanged, for macro
  signals normalized over own history via EXPANDING z-score).
- **Blend roster (4): TSMomentum, BondCarry, FXCarry, InflationTrend.**
  GrowthTrend DROPPED (COVID publication-lag whipsaw; code retained; reintroduce
  with a Phase 3 nowcasting layer). BondCarry is the strongest signal (IC 0.039
  @1d, t=4.9); InflationTrend is regime-conditional (IC ~0 full-sample, +0.124
  @21d in 2021–23 inflation; IC negatively correlated with BondCarry → ballast).
  BondCarry needs `dividends` + `prices_unadjusted` injected via constructor
  (dist yield = trailing dividends / UNADJUSTED price).

### Phase 4 — what was built and the key decisions

**`portfolio/blend.py` — `blend_signals(signals, weights=None, scales=None)`.**
Per-date, per-asset weighted average over the signals PRESENT (non-NaN) at each
cell, weights renormalized over the present set (absent signal adds 0 to
numerator AND denominator — the .sum()-drops-NaN fix). Union grid over
dates/assets (never intersection). `scales` divides each signal by a constant
first. Defaults: equal weights. Denominator poisoned before dividing
(`present_weight.where(>0)`).

**`portfolio/roster.py` — `build_roster(...)` + `ROSTER_SCALES`.**
Single source of truth for signal construction (BondCarry injection lives ONCE).
`ROSTER_SCALES = {"InflationTrend": 2.0}`: time-series-normalized InflationTrend
reaches ±4.5 (expanding z vs own history; 2022 was a genuine ~4σ regime) while
cross-sectional signals sit ±2; dividing by an A-PRIORI constant 2.0 (NOT fitted
— no sweeping c) preserves shape (regime info) while preventing inflation from
hijacking the equal-weight blend. The shape-preserving rescale IS the
regime-conditional weighting: quiet in calm, loudest voice in regimes, no
explicit gate. (User originally wanted un-rescaled "inflation-dominated crash
protection"; was argued out of it: inflation regimes ≠ most crashes — momentum
is the general crisis hedge; rescaled keeps inflation loudest anyway.)

**`portfolio/sizing.py` — `compute_target_weights(composite, prices, mode,
target_vol=0.10, vol_window=126)`.**
Stage A: mode fork. `long_short` = composite straight through, **NO demean**
(net exposure retained; the only net-exposure source is InflationTrend's betas
summing to ~−4, so the net tilt IS the inflation view: net short risk/duration
when inflation rises — deliberate, see DECISIONS; demean is a one-line switch to
flip with attribution evidence). `long_only` = clip shorts to 0.
Stage B: inverse-vol, w ∝ dir/σ (126d trailing, `realized_volatility` reused).
Intermediate weights are unnormalized — only ratios matter.
Stage C: per-date scalar k_t = target / √(wᵀΣw·252), Σ = trailing 126d rolling
covariance, CURRENT weights on PAST returns (PIT). Implemented vectorized via
the rolling-cov MultiIndex panel + two contractions (broadcast weights over the
asset_i level → Σw → unstack → wᵀΣw). Explicit covariance chosen over
matrix-free (DECISIONS: noise in Σ matters when OPTIMIZING against it, not when
MEASURING a scalar). Output: daily, gross-UNCONSTRAINED.

**`portfolio/risk.py` — `apply_risk_limits(weights, risk_buckets,
max_position=0.20, max_bucket=0.50, max_gross=3.0)`.**
Caps are fractions of **TARGET gross** → fixed ABSOLUTE thresholds (0.60 /
1.50 / 3.0) — the **B2 interpretation**. (First implementation used fractions
of realized gross; the diagnostic caught it as infeasible/self-referential on
concentrated books — long-only single-name days showed post-cap "1.000". Key
gotcha §8.) Proportional scaling for bucket & leverage caps (preserves the
signal's relative view); clip for single names (clip == scale for one position).
Order per-asset → bucket → leverage, once each; leverage last is EXACT under
absolute caps (uniform downscale can't re-breach a fixed threshold). Buckets +
limits live in `universe.yaml` (`risk_buckets` regroups by RISK DRIVER: LQD/EMB
→ credit, BWX → rates). Coverage guard raises on unbucketed/double-bucketed
assets. Drawdown throttle deliberately NOT here (path-dependent → engine).

**`backtest/engine.py` — `run_backtest(weights, prices, rf, rebalance="ME",
cost_per_side=0.0001)`.**
THE place the execution lag lives (nowhere else): targets read at rebalance t
earn returns from t+1 (statement order in the daily loop enforces it). Drift
between monthly rebalances: w ← w(1+r)/(1+r_p); turnover measured against the
DRIFTED book. Costs 1bp/side on one-way traded notional. Cash/financing:
r_p = Σw_i r_i + (1−net)·rf, rf = DGS3MO/100/252 (note: letter O, not zero).
**FLAGGED simplification: same-rate financing, no borrow fees — flatters
long-short.** No drawdown throttle in v1 (deferred overlay). Plain Python loop
(drift is sequential; clarity over vectorization).

**`backtest/performance.py`** — pure functions of a daily return series:
cagr, annualized_vol, sharpe & sortino (**EXCESS over rf**; sortino denominator
= full-sample RMS of negative excess), max_drawdown (equity curve, negative),
calmar (NaN-guarded), time_underwater, hit_rate, `summarize()`. Contract: caller
passes the LIVE slice (engine returns include flat pre-live zeros). No benchmark
logic in the module — comparisons are a script's job.

---

## 4. First full-stack results (run_backtest.py, 1bp/side, 2004-02→2026-05)

| | long_short | long_only |
|---|---|---|
| Realized ann vol (target 0.10) | 0.1063 | 0.1161 |
| CAGR | 5.92% | 7.87% |
| Sharpe (excess) | 0.43 | 0.56 |
| Sortino | 0.59 | 0.75 |
| MaxDD | −21.5% | −32.8% |
| Calmar | 0.28 | 0.24 |
| Time underwater | 860d | 495d |
| Hit rate | 53.4% | 55.2% |
| Ann turnover (gross) | 10.1x | 5.9x |
| Cost drag | 10 bps/y | 6 bps/y |

**How to read this honestly (the assistant's standing interpretation):**
- Sharpes in the 0.4–0.7 "honest retail" band — NOT suspiciously strong. No
  leakage hunt indicated. Sortino > Sharpe both modes (no downside-skew poison).
- **Do NOT conclude "long-only wins" from raw Sharpe.** Long-only runs ~1.46 net
  long; much of its 0.56 is the market risk premium (beta anyone can buy).
  Long-short (net ~+0.46, often lower) is much closer to pure alpha. The fair
  fight = beta decomposition (§7, FIRST task). Long-short's defensive shape
  already shows: MaxDD −21.5% vs −32.8%.
- Vol OVERSHOOTS target (esp. long-only): trailing-vol targeting lags vol
  spikes (clustering); long-only worse because net-long books concentrate in
  the common risk-on factor where correlation spikes bite. Normal (±20% band);
  re-examine at attribution.
- Turnover ~4–4.4 full book turns/yr — higher than signal speed implies because
  the SIZING layer trades too (k_t rescales whole book monthly; leverage cap
  toggles). Decomposition deferred; if scalar dominates, smooth k_t.
- 860d underwater = 3.5 years. Normal for systematic macro but decision-relevant;
  locate WHICH years at regime-slicing.
- Same-rate financing flatters long-short specifically.

---

## 5. Phase 4 commits (chronological)

c6788c6 fix InflationTrend object dtype (pd.NA → np.nan) · d3ed54b blend ·
5ee67cc roster · adc6fea sizing · f698e9c risk (B2) · 1375efc engine ·
e70daa9 engine DECISIONS · f6da955 performance. All pushed.

---

## 6. Full deferred-items list

1. Commodity roll-yield carry — needs futures-curve data (not free). Deferred.
2. Graded/estimated macro betas — Phase 3 revisit; magnitude lives in sizing.
3. **GrowthTrend reintroduction** — once Phase 3 nowcasting beats the pub lag.
4. **ML nowcasting layer** (`models/`, Phase 3) — the one valuable ML use here;
   also the GrowthTrend trigger. Direct ML return prediction = learning
   baseline only.
5. **Financing-spread parameter** + short borrow fees (engine; flagged).
6. **Drawdown throttle** as optional engine overlay (own design pass).
7. **Turnover decomposition** (signal vs vol-scalar vs cap toggling); smooth
   k_t if scalar dominates.
8. **Vol-overshoot examination** (esp. long-only correlation-spike exposure).
9. **Demean check at attribution**: net-tilt vol vs RV vol; flip demean only
   with evidence.
10. **Sensitivity sweeps**: risk-cap values, cost/side, vol window, rebalance
    frequency ("small changes → small changes").
11. Docstring-condensing pass on blend.py / sizing.py to the lean bar.
12. IC-weighted blend weights (blend already parameterized; 1/N default kept).

---

## 7. NEXT: the results conversation (agenda, in priority order)

1. **Beta decomposition** — regress each mode's daily EXCESS returns on market
   excess returns (and vs a 60/40 benchmark); compare ALPHAS, not raw Sharpes.
   This settles long-short vs long-only honestly. START HERE.
2. **Regime slices** — GFC / 2010s calm / COVID / 2021–23 inflation / 2024–26:
   did the inflation tilt protect in 2022? (The rescale + no-demean thread gets
   its verdict.) Where do the 860 underwater days live?
3. **Attribution** — per-signal & per-bucket contribution; turnover decomposition;
   net-tilt vs RV vol (the demean question).
4. **Sensitivity sweeps** (item 10).
Then: README/Phase-4 writeup, and Phase 3 (nowcasting) or live/ scaffolding.

---

## 8. Hard-won gotchas (Phase 1–2 list + Phase 4 additions — avoid repeating)

- **Vintage value-selection**: per vintage take the LATEST observation_date
  (sort by tiebreaker before first/last). A dedup bug here once injected
  1940s values onto 2003 dates. Fixed + regression-tested.
- **Publication lag can INVERT a macro signal in fast shocks** (killed
  GrowthTrend in COVID). Nowcasting is the fix.
- **Full-sample IC masks regime-conditional signals** (saved InflationTrend).
- **Per-date weight renormalization** for any weighted combo of components with
  different start dates (divide by PRESENT weight, never fixed N).
- **Look-ahead acid test**: feed future info; IC must DEGRADE.
- **Test the invariant, not a proxy** (bound test caught the NaN-weighting bug).
- **Distribution yield denominator = UNADJUSTED price.**
- **NEW — `pd.NA` poisons dtype**: one `replace(0.0, pd.NA)` made InflationTrend
  object-dtype; object 0/0 divides at PYTHON level and raises ZeroDivisionError
  (numpy float64 0/0 = nan + warning). Contract: float64 + np.nan everywhere.
  The `<NA>` print marker is the tell.
- **NEW — poison denominators BEFORE dividing**: `denom.where(denom > 0)` then
  divide (x/NaN = NaN, never raises). Used in blend, sizing, calmar.
- **NEW — fraction-of-REALIZED-gross caps are self-referential and infeasible**
  on concentrated books (one name = 100% of own gross forever). Cap against a
  STABLE denominator (target gross → absolute thresholds). Diagnostics on real
  data caught this where unit tests wouldn't have — always run the real-data
  diagnostic BEFORE writing tests.
- **NEW — backtest lag lives in exactly ONE place** (engine statement order).
  Blend/sizing/risk never shift; double-lagging is the bug to avoid.
- **NEW — turnover must be measured against the DRIFTED book** (stale-target
  turnover understates costs; constant-weights-between-rebalances is fictional).
- **NEW — raw Sharpe comparisons across books with different net exposure are
  beta-confounded.** Decompose before concluding.

---

## 9. Suggested first message to the new session

Upload this file and say: "This is the handover for my atlas quant project,
Phases 1–4 complete (signals through costed backtest; results table in §4). I
want to start the results conversation — beta decomposition first, per §7.
Conceptual design first, as usual." The assistant should confirm the
separate-sandbox workflow and the explain-everything expectation before starting.
