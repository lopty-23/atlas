# Design Decisions

Non-obvious judgment calls, with the evidence behind them. Newest first.

## Phase 2 — Signal roster (final: 4 of 5 signals)

After IC evaluation (Stage 2.5) and correlation/regime analysis (Stage 2.7),
the blend roster is: **TSMomentum, BondCarry, FXCarry, InflationTrend.**

**GrowthTrend — DROPPED from the blend, code retained.**
- Standalone IC weak (full-sample 0.018 @21d, never significant, t<2 everywhere).
- Regime check: positive in GFC (+0.05/+0.08) and calm (+0.05/+0.12), but
  STRONGLY NEGATIVE in COVID (-0.145 @21d, -0.384 @126d).
- The COVID failure is a structural publication-lag whipsaw: the growth
  collapse and V-recovery both happened inside the data-publication lag, so the
  signal went maximally defensive (on just-published bad data) right as risk
  assets bottomed and ripped. Fast shocks will whipsaw it the same way.
- Also correlated with TSMomentum (IC-corr 0.25-0.35) — redundant with the
  stronger momentum signal in the regimes where it IS right.
- Verdict: redundant when right, structurally harmful when wrong. Worst combo.
- REINTRODUCE once a Phase 3 nowcasting layer addresses the publication lag
  (nowcasting would catch the collapse before the lagged data prints). Code and
  tests remain in src/atlas/signals/macro_trend.py.

**InflationTrend — KEPT, despite ~0 full-sample IC.**
- Full-sample IC slightly negative (-0.016 @21d) but NOT significant (t~-0.8) —
  i.e. zero edge, not negative edge.
- Regime check: +0.08 IC in high-inflation 2021-2023, +0.02 in calm — a genuine
  regime-conditional edge masked by full-sample averaging (the quiescent 2010s
  and the 2024 disinflation drag the average to ~0).
- IC-correlation with BondCarry is negative (-0.16 @126d): provides ballast when
  the dominant carry signal struggles (rising inflation hurts bonds).
- It is a CONDITIONAL signal (inflation-regime specialist), best used with
  regime awareness rather than a constant weight — flag for Phase 4 blend.

**Lesson:** full-sample IC masks regime-conditional signals. Both macro signals
looked weak/dead on full-sample IC; regime splits revealed inflation has a real
conditional edge (kept) and growth has a structural failure mode (dropped).
Evaluate macro signals regime-conditionally, not just full-sample.

## Phase 2 — Signal tuning (Stage 2.5 follow-up)

**Momentum (lookback, skip_days) — defaults CONFIRMED, no change.**
- Tested lookbacks {21, 63, 126, 252} × skip_days {0, 5, 21} via IC decay.
- Longer lookbacks clearly predict longer horizons better (126/252 dominate
  21/63 at the 126d/252d horizons); 21-day lookback is weakest throughout.
  Pattern is robust across all skip values. Confirms lookback=252 default.
- skip_days has negligible effect (252-lookback/252-horizon IC: 0.091/0.092/
  0.093 for skip 0/5/21) — short-term reversal is weak in broad ETF baskets,
  so the single-stock "skip recent month" convention doesn't transfer. Keep
  skip_days=0.

**InflationTrend — breakeven up-weighted to 50% (was equal-weight 25%).**
- Component IC isolation: breakeven_only (T5YIE) high-inflation IC @21d = 0.140
  vs realized_only (CPI/coreCPI/PCE) = 0.065. Breakeven is also the only
  component net-positive full-sample. Forward-looking market expectations
  predict far better than backward-looking, lagged realized prints.
- Chose breakeven_weight=0.5 (Option A: up-weight, retain realized as a
  cross-regime diversifier) over breakeven-only, to avoid over-fitting to the
  single 2021-2023 inflation episode.
- Fixed a NaN-handling bug found during testing: weighted combination now
  renormalizes per-date over present (non-NaN) components, so a component in
  its expanding-zscore warm-up doesn't silently halve the composite. This
  per-date weight-renormalization pattern applies to ANY combination of
  components with different start dates — relevant to Phase 4 signal blending.

## Phase 4 — Portfolio vol-targeting (Step C) uses an explicit rolling covariance

**Decision.** Estimate the portfolio's ex-ante vol for vol-targeting via an
explicit trailing rolling covariance (`returns.rolling(126).cov()`), contracted
against the current weights as w^T Sigma w, vectorized (no per-date loop).
Window = 126 (same as inverse-vol). Target = 10% annualized.

**Alternative considered.** A "matrix-free" form: synthesize the portfolio
return stream by applying current weights to past returns, take its rolling std.
Computes the identical number without materializing Sigma.

**Rationale.** Both give the same w^T Sigma w. The matrix-free form was initially
preferred to avoid estimating a noisy 21x21 covariance. On reflection that
argument was overstated *for this use*: covariance-estimation noise is harmful
when you OPTIMIZE against Sigma (off-diagonal errors push weights around), but
Step C only MEASURES a scalar (today's book vol) -- it never inverts or optimizes
Sigma. For a pure measurement the matrix is harmless, and the explicit-covariance
code is clearer and less error-prone than the fixed-weights-rolling-std
contortion. At 21 assets the matrix carries no performance cost.

The anti-noise prior still correctly governs SIZING (inverse-vol over full ERC,
per DeMiguel); it simply does not bite on a measurement.

**Point-in-time.** Covariance uses only trailing returns; weights are current.
The execution lag is applied later (backtest), never here.

## Phase 4 — Long-short book is NOT demeaned (net inflation tilt retained)

**Decision.** In `sizing.py` Stage A, the long_short composite passes through
unchanged -- we do NOT cross-sectionally demean it to force dollar-neutrality.
The book therefore carries whatever net long/short exposure the composite
implies.

**Where the net exposure comes from.** Three of the four signals (TSMomentum,
BondCarry, FXCarry) are cross-sectionally z-scored, so each is mean-zero across
its assets every day and contributes ~zero net exposure by construction. The
ONLY net-exposure source is InflationTrend: it is beta-mapped (not z-scored) and
its betas sum to ~-4 across the universe (+1 on six assets, -1 on ten, 0 on
five). So the book's net tilt is, in effect, the inflation signal's directional
macro view: net short risk-assets-and-duration when inflation is rising, net
long when disinflating.

**Rationale.** Demeaning would strip out exactly that view -- and the inflation
signal leaning the book net-short in an inflation shock IS the protection we
chose to keep (consistent with retaining InflationTrend at all, and with the
shape-preserving rescale rather than re-z-scoring). The tilt is modest on
average and economically coherent, not an artifact.

**Empirical check (build_sizing.py, long_short).** Net exposure mean +0.46,
ranging -2.15 (inflationary stretches, net short) to +1.98. Near-neutral on
average with the expected negative excursions when inflation is live -- the
designed behavior, made visible.

**Flagged for attribution.** A net directional tilt contributes more vol per
unit gross than an offsetting relative-value spread, so the tilt may punch above
its blend-weight share in inflation regimes. Whether it stays modest or starts
dominating book RISK is an empirical question for performance attribution
(decompose net-exposure vol vs relative-value vol). If it dominates, demeaning
is a one-line switch to flip -- WITH evidence, not pre-emptively.

## Phase 4 — Risk limit values (literature-anchored anti-domination backstops)

**Decision.** risk.py applies three static caps to sizing's gross-unconstrained
weights. Per-asset and per-bucket caps are fractions of TARGET gross
(max_gross), making them FIXED ABSOLUTE thresholds independent of the book's
realized gross or concentration:

| Cap          | Value | Absolute (x max_gross) | Role                          |
|--------------|-------|------------------------|-------------------------------|
| max_position | 0.20  | 0.60                   | No single asset dominates     |
| max_bucket   | 0.50  | 1.50                   | No risk bucket dominates      |
| max_gross    | 3.0   | (itself)               | Gross leverage tail backstop  |

Enforced by PROPORTIONAL SCALING for the bucket and leverage caps, preserving
the signal's relative view within a bucket / across the book -- caps control the
SIZE of a bet, not WHICH bet. Single-name breaches of max_position are trimmed
(clip == scale for one position). Order: per-asset -> bucket -> leverage,
applied once each. Leverage last is exact, not approximate: uniform whole-book
scaling only REDUCES each |w|, and reducing a value already at-or-under a fixed
absolute cap cannot push it back over. All caps only reduce; freed capital goes
to cash (book may run under its vol target on capped days), never into uncapped
names (which would re-concentrate).

**Why fractions of TARGET gross, not realized gross (the B2 choice).** The
first implementation capped against realized gross (sum|w| of the day's book).
The build_risk.py diagnostic caught this as broken on real data: a
fraction-of-realized-gross cap is SELF-REFERENTIAL and infeasible on
concentrated books -- on a long-only defensive day the book can collapse to one
surviving name, which is 100% of its own gross no matter how much you trim it
(post-cap max position read 1.000, the cap silently doing nothing exactly when
the book was most concentrated). Capping against the fixed target gross makes
the thresholds absolute (0.60 / 1.50), always feasible, and stable: "never bet
more than X of intended capital on one name," where the reference does not move
when you trim. Realized-gross fractions in diagnostics remain informational
only; the enforced quantity is absolute weight.

**Rationale for the values.** All three are anti-domination backstops, not
active position shapers -- the vol-target (sizing) controls expected risk; these
bound TAIL/concentration risk when the trailing-vol estimate is wrong. Values
chosen to "let the signal play out as much as possible": per-asset (0.20) and
per-bucket (0.50) are deliberately loose so they rarely bind on the diversified
~19-asset inverse-vol book; only the leverage cap is set to actively bind.

Per-bucket is UNIFORM (0.50 for every bucket), not bespoke per class -- bespoke
caps would encode a view on which classes "deserve" more room, an overfitting
trap. If attribution later shows one bucket needs a tighter leash, tighten THAT
one with evidence.

**Leverage cap (3.0) -- why this value.** Sizing's gross (build_sizing.py,
long_short) runs mean 2.58, p95 4.14, max 6.07. 3.0 leaves the median book
(2.55) untouched, lightly trims the upper quartile, and hard-stops the dangerous
4-6x tail -- the LOW-trailing-vol-estimate days the cap exists to catch (the
failure that blew up naive risk-parity in Mar-2020 and managed futures in 2022).
A 2.0 cap would bind >50% of days and override the vol-target (making the
leverage cap the de-facto sizer); 5.0+ would never catch the tail. Verified on
real data (build_risk.py): binds 33.9% of long-short days, 1.7% long-only;
post-cap absolute max position 0.600 and max bucket 1.500, exact in both modes.

**Literature anchors.** Per-asset ~10-20% and per-class ~50% are standard
diversified-mandate / balanced-fund conventions (the latter a tightening vs
60/40's 60% equity); gross 2-3x is the classic risk-parity / managed-futures
band for hitting ~10% vol across low-vol assets (Bridgewater All Weather, AQR).
The leverage-cap-as-tail-backstop framing follows Lopez de Prado.

**Flagged for sensitivity check.** These are PRIORS from literature, NOT fitted
to atlas. They WILL shape results -- the 3.0 leverage cap disproportionately
trims the highest-gross days, which are the inflation-tilt days (net exposure to
-2.15). Sensitivity-sweep all three in the backtest ("small parameter changes ->
small performance changes" robustness, per the research doc); all three live in
universe.yaml (risk_limits) for trivial tuning.

## Phase 4 — Backtest engine conventions and flagged simplifications

**Decision.** engine.py simulates daily target weights with: execution lag
(targets read at rebalance t earn returns from t+1 -- the lag lives HERE and
nowhere else; blend/sizing/risk never shift), monthly rebalance (last trading
day per month), drift between rebalances (positions held, weights move with
relative returns: w_i <- w_i(1+r_i)/(1+r_p); turnover measured against the
DRIFTED book, not stale targets), costs at 1bp per side on one-way traded
notional (~2bp round-trip, mid-range of the 1-3bp ETF band; parameterized),
and a cash/financing leg: r_p = sum(w_i r_i) + (1 - net) * rf, rf = DGS3MO
(point-in-time, /100/252).

**FLAGGED simplification -- same-rate financing, no borrow fees.** The cash leg
lends AND borrows at rf, with no short-borrow fees. Research convention, but it
flatters the long-short book: real margin costs rf-plus-spread, and shorts in
HYG/EMB/EEM carry borrow fees. Long-short results lean optimistic on financing.
A financing-spread parameter is an easy later refinement; revisit before any
live-relevance claim.

**No drawdown throttle in v1.** Path-dependent (needs the running equity
curve), so it could not live in risk.py; deliberately EXCLUDED from engine v1 to
keep the core loop's testing surface clean and to establish the unconditional
baseline a throttle would be judged against. Deferred as an optional overlay
with its own design pass (throttle level / action are free parameters).

**Empirical flags from the first full run (run_backtest.py, 1bp costs):**
- Realized vol OVERSHOOTS target: 0.1063 long-short, 0.1161 long-only vs 0.10.
  Trailing-window vol-targeting under-estimates forward vol when vol spikes
  (vol clusters; the 126d estimate lags, monthly rebalance adds staleness) --
  this beat the opposing effect (leverage cap trimming). Long-only worse
  because a net-long book concentrates in the common risk-on factor, where
  correlation spikes bite hardest. Within the normal +-20% band for this
  machinery; re-examine at the performance/attribution stage.
- Turnover ~10.1x gross/yr long-short, ~5.9x long-only (~4-4.4 full book turns
  /yr, holding ~3 months). Higher than signal speed alone implies: the
  vol-target scalar k_t rescales the WHOLE book monthly, the leverage cap
  toggles, and inverse-vol weights shift -- sizing-layer turnover on top of
  signal turnover. Cost-trivial at 1bp (~10 bps/yr drag) but DEFERRED:
  decompose turnover (signal vs vol-scalar vs cap toggling) at the attribution
  stage; if the scalar dominates, smooth k_t (e.g. re-scale only on >x% moves).