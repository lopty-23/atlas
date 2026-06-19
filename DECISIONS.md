# Design Decisions

Non-obvious judgment calls, with the evidence behind them. Newest first.

## Phase 2 — Signal roster (final: 4 of 5 signals)

After IC evaluation (Stage 2.5) and correlation/regime analysis (Stage 2.7), the
blend roster is **TSMomentum, BondCarry, FXCarry, InflationTrend**.

**GrowthTrend — dropped from the blend, code retained.**
- Standalone IC weak (full-sample 0.018 @21d, never significant, t < 2 everywhere).
- Regime check: positive in GFC (+0.05/+0.08) and calm (+0.05/+0.12), but
  strongly negative in COVID (-0.145 @21d, -0.384 @126d).
- The COVID failure is a structural publication-lag whipsaw: the growth collapse
  and V-recovery both happened inside the data-publication lag, so the signal went
  maximally defensive on just-published bad data right as risk assets bottomed and
  ripped. Fast shocks will whipsaw it the same way.
- Also correlated with TSMomentum (IC-corr 0.25-0.35), so it's redundant with the
  stronger momentum signal in the regimes where it is right.
- Verdict: redundant when right, harmful when wrong. Reintroduce once a Phase 3
  nowcasting layer addresses the publication lag (nowcasting would catch the
  collapse before the lagged data prints). Code and tests remain in
  src/atlas/signals/macro_trend.py.

**InflationTrend — kept, despite ~0 full-sample IC.**
- Full-sample IC slightly negative (-0.016 @21d) but not significant (t ~ -0.8),
  i.e. zero edge, not negative edge.
- Regime check: +0.08 IC in high-inflation 2021-2023, +0.02 in calm — a real
  regime-conditional edge masked by full-sample averaging (the quiet 2010s and
  the 2024 disinflation drag the average to ~0).
- IC-correlation with BondCarry is negative (-0.16 @126d): it provides ballast
  when the dominant carry signal struggles (rising inflation hurts bonds).
- A conditional signal (inflation-regime specialist), best used with regime
  awareness rather than a constant weight. Flagged for the Phase 4 blend.

**Lesson.** Full-sample IC masks regime-conditional signals. Both macro signals
looked weak on full-sample IC; regime splits revealed inflation has a real
conditional edge (kept) and growth has a structural failure mode (dropped).
Evaluate macro signals regime-conditionally, not just full-sample.

## Phase 2 — Signal tuning (Stage 2.5 follow-up)

**Momentum (lookback, skip_days) — defaults confirmed, no change.**
- Tested lookbacks {21, 63, 126, 252} x skip_days {0, 5, 21} via IC decay.
- Longer lookbacks predict longer horizons better (126/252 dominate 21/63 at the
  126d/252d horizons); the 21-day lookback is weakest throughout, robust across
  all skip values. Confirms lookback=252.
- skip_days has negligible effect (252-lookback/252-horizon IC: 0.091/0.092/0.093
  for skip 0/5/21). Short-term reversal is weak in broad ETF baskets, so the
  single-stock "skip recent month" convention doesn't transfer. Keep skip_days=0.

**InflationTrend — breakeven up-weighted to 50% (was equal-weight 25%).**
- Component IC isolation: breakeven_only (T5YIE) high-inflation IC @21d = 0.140 vs
  realized_only (CPI/coreCPI/PCE) = 0.065. The breakeven is also the only
  component net-positive full-sample. Forward-looking market expectations predict
  far better than backward-looking, lagged realized prints.
- Chose breakeven_weight=0.5 (up-weight, retain realized as a cross-regime
  diversifier) over breakeven-only, to avoid over-fitting to the single 2021-2023
  inflation episode.
- Fixed a NaN-handling bug found during testing: the weighted combination now
  renormalizes per-date over present (non-NaN) components, so a component in its
  expanding-zscore warm-up doesn't silently halve the composite. This per-date
  renormalization applies to any combination of components with different start
  dates — relevant to Phase 4 signal blending.

## Phase 4 — Vol-targeting (Step C) uses an explicit rolling covariance

**Decision.** Estimate the portfolio's ex-ante vol via an explicit trailing
rolling covariance (`returns.rolling(126).cov()`), contracted against the current
weights as w^T Sigma w, vectorized (no per-date loop). Window = 126 (same as
inverse-vol). Target = 10% annualized.

**Alternative considered.** A matrix-free form: synthesize the portfolio return
stream by applying current weights to past returns, take its rolling std. Same
number without materializing Sigma.

**Rationale.** Both give the same w^T Sigma w. The matrix-free form was initially
preferred to avoid estimating a noisy 21x21 covariance, but that argument is
overstated for this use: covariance-estimation noise hurts when you optimize
against Sigma (off-diagonal errors push weights around), but Step C only measures
a scalar (today's book vol); it never inverts or optimizes Sigma. For a pure
measurement the matrix is harmless, and the explicit-covariance code is clearer
than the fixed-weights-rolling-std version. At 21 assets the matrix carries no
performance cost. The anti-noise prior still correctly governs sizing (inverse-vol
over full ERC, per DeMiguel); it just doesn't bite on a measurement.

**Point-in-time.** Covariance uses only trailing returns; weights are current.
The execution lag is applied later (backtest), never here.

## Phase 4 — Long-short book is not demeaned (net inflation tilt retained)

**Decision.** In sizing.py Stage A, the long_short composite passes through
unchanged — it is not cross-sectionally demeaned to force dollar-neutrality. The
book carries whatever net long/short exposure the composite implies.

**Where the net exposure comes from.** Three of the four signals (TSMomentum,
BondCarry, FXCarry) are cross-sectionally z-scored, so each is mean-zero across
its assets every day and contributes ~zero net exposure by construction. The only
net-exposure source is InflationTrend: it is beta-mapped (not z-scored) and its
betas sum to ~-4 across the universe (+1 on six assets, -1 on ten, 0 on five). So
the book's net tilt is the inflation signal's directional view: net short
risk-assets-and-duration when inflation is rising, net long when disinflating.

**Rationale.** Demeaning would strip out exactly that view, and the inflation
signal leaning the book net-short in an inflation shock is the protection we chose
to keep (consistent with retaining InflationTrend at all, and with the
shape-preserving rescale rather than re-z-scoring). The tilt is modest on average
and economically coherent, not an artifact.

**Empirical check (build_sizing.py, long_short).** Net exposure mean +0.46,
ranging -2.15 (inflationary stretches, net short) to +1.98. Near-neutral on
average with the expected negative excursions when inflation is live — the
designed behavior, made visible.

**Flagged for attribution.** A net directional tilt contributes more vol per unit
gross than an offsetting relative-value spread, so the tilt may punch above its
blend-weight share in inflation regimes. Whether it stays modest or starts
dominating book risk is an empirical question for attribution (decompose
net-exposure vol vs relative-value vol). If it dominates, demeaning is a one-line
switch to flip — with evidence, not pre-emptively.

## Phase 4 — Risk limit values (literature-anchored backstops)

**Decision.** risk.py applies three static caps to sizing's gross-unconstrained
weights. Per-asset and per-bucket caps are fractions of target gross (max_gross),
making them fixed absolute thresholds independent of the book's realized gross or
concentration:

| Cap          | Value | Absolute (x max_gross) | Role                          |
|--------------|-------|------------------------|-------------------------------|
| max_position | 0.20  | 0.60                   | No single asset dominates     |
| max_bucket   | 0.50  | 1.50                   | No risk bucket dominates      |
| max_gross    | 3.0   | (itself)               | Gross leverage tail backstop  |

Enforced by proportional scaling for the bucket and leverage caps, preserving the
signal's relative view within a bucket and across the book — caps control the size
of a bet, not which bet. Single-name breaches of max_position are trimmed (clip ==
scale for one position). Order: per-asset, then bucket, then leverage, applied
once each. Leverage last is exact, not approximate: uniform whole-book scaling
only reduces each |w|, and reducing a value already at-or-under a fixed absolute
cap cannot push it back over. All caps only reduce; freed capital goes to cash
(the book may run under its vol target on capped days), never into uncapped names
(which would re-concentrate).

**Why fractions of target gross, not realized gross (the B2 choice).** The first
implementation capped against realized gross (sum|w| of the day's book). The
build_risk.py diagnostic caught this as broken on real data: a
fraction-of-realized-gross cap is self-referential and infeasible on concentrated
books — on a long-only defensive day the book can collapse to one surviving name,
which is 100% of its own gross no matter how much you trim it (post-cap max
position read 1.000, the cap silently doing nothing exactly when the book was most
concentrated). Capping against the fixed target gross makes the thresholds
absolute (0.60 / 1.50), always feasible, and stable: never bet more than X of
intended capital on one name, where the reference doesn't move when you trim.
Realized-gross fractions in diagnostics remain informational only; the enforced
quantity is absolute weight.

**Rationale for the values.** All three are backstops, not active position
shapers — the vol-target (sizing) controls expected risk; these bound
tail/concentration risk when the trailing-vol estimate is wrong. Values chosen to
let the signal play out as much as possible: per-asset (0.20) and per-bucket
(0.50) are deliberately loose so they rarely bind on the diversified ~19-asset
inverse-vol book; only the leverage cap is set to actively bind. Per-bucket is
uniform (0.50 for every bucket), not bespoke per class — bespoke caps would encode
a view on which classes deserve more room, an overfitting trap. If attribution
later shows one bucket needs a tighter leash, tighten that one with evidence.

**Leverage cap (3.0) — why this value.** Sizing's gross (build_sizing.py,
long_short) runs mean 2.58, p95 4.14, max 6.07. 3.0 leaves the median book (2.55)
untouched, lightly trims the upper quartile, and hard-stops the dangerous 4-6x
tail — the low-trailing-vol-estimate days the cap exists to catch (the failure
that blew up naive risk-parity in Mar-2020 and managed futures in 2022). A 2.0 cap
would bind >50% of days and override the vol-target (making the leverage cap the
de-facto sizer); 5.0+ would never catch the tail. Verified on real data
(build_risk.py): binds 33.9% of long-short days, 1.7% long-only; post-cap absolute
max position 0.600 and max bucket 1.500, exact in both modes.

**Literature anchors.** Per-asset ~10-20% and per-class ~50% are standard
diversified-mandate / balanced-fund conventions (the latter a tightening vs
60/40's 60% equity); gross 2-3x is the classic risk-parity / managed-futures band
for hitting ~10% vol across low-vol assets (Bridgewater All Weather, AQR). The
leverage-cap-as-tail-backstop framing follows Lopez de Prado.

**Flagged for sensitivity check.** These are priors from literature, not fitted to
atlas. They will shape results — the 3.0 leverage cap disproportionately trims the
highest-gross days, which are the inflation-tilt days (net exposure to -2.15).
Sensitivity-sweep all three in the backtest; all three live in universe.yaml
(risk_limits) for trivial tuning.

## Phase 4 — Backtest engine conventions and flagged simplifications

**Decision.** engine.py simulates daily target weights with: execution lag
(targets read at rebalance t earn returns from t+1 — the lag lives here and
nowhere else; blend/sizing/risk never shift), monthly rebalance (last trading day
per month), drift between rebalances (positions held, weights move with relative
returns: w_i <- w_i(1+r_i)/(1+r_p); turnover measured against the drifted book,
not stale targets), costs at 1bp per side on one-way traded notional (~2bp
round-trip, mid-range of the 1-3bp ETF band; parameterized), and a cash/financing
leg: r_p = sum(w_i r_i) + (1 - net) * rf, rf = DGS3MO (point-in-time, /100/252).

**Flagged simplification — same-rate financing, no borrow fees.** The cash leg
lends and borrows at rf, with no short-borrow fees. Research convention, but it
flatters the long-short book: real margin costs rf-plus-spread, and shorts in
HYG/EMB/EEM carry borrow fees. Long-short results lean optimistic on financing. A
financing-spread parameter is an easy later refinement; revisit before any
live-relevance claim.

**No drawdown throttle in v1.** Path-dependent (needs the running equity curve),
so it couldn't live in risk.py; deliberately excluded from engine v1 to keep the
core loop's testing surface clean and to establish the unconditional baseline a
throttle would be judged against. Deferred as an optional overlay with its own
design pass (throttle level / action are free parameters).

**Empirical flags from the first full run (run_backtest.py, 1bp costs):**
- Realized vol overshoots target: 0.106 long-short, 0.116 long-only vs 0.10.
  Trailing-window vol-targeting under-estimates forward vol when vol spikes (vol
  clusters; the 126d estimate lags, monthly rebalance adds staleness), and this
  beats the opposing effect (leverage cap trimming). Long-only is worse because a
  net-long book concentrates in the common risk-on factor, where correlation
  spikes bite hardest. Within the normal +-20% band; re-examine at attribution.
- Turnover ~10.1x gross/yr long-short, ~5.9x long-only (~4-4.4 full book turns/yr,
  holding ~3 months). Higher than signal speed alone implies: the vol-target
  scalar k_t rescales the whole book monthly, the leverage cap toggles, and
  inverse-vol weights shift — sizing-layer turnover on top of signal turnover.
  Cost-trivial at 1bp (~10 bps/yr drag). Deferred: decompose turnover at the
  attribution stage; if the scalar dominates, smooth k_t.

## Phase 4 Results — Beta decomposition + regime slices (the verdict)

**Method.** scripts/evaluate_backtest.py: single-factor regression of each mode's
daily excess returns on SPY excess returns; alpha annualized, HAC t-stat
(Newey-West, 21 lags) conditioning on estimated beta. scripts/evaluate_regimes.py:
sub-window metrics on four macro regimes (windows fixed on public events before
seeing results) plus a diversifier test (long_short overlaid on 60/40). 60/40
(SPY/IEF) run through the same engine for apples-to-apples. Regression validated by
the 60/40 anchor: beta 0.55, R2 0.94 (a 60% SPY book mechanically has beta ~0.6,
confirming the machinery).

**Stage 1 — the raw-Sharpe ranking inverts under decomposition.**

| | Sharpe | alpha/y | alpha t | beta | R2 | MaxDD |
|---|--------|---------|---------|------|----|-------|
| long_short | 0.43 | +3.55% | 1.65 | 0.10 | 0.03 | -21.5% |
| long_only  | 0.55 | +2.81% | 1.41 | 0.35 | 0.32 | -32.8% |
| 60/40      | 0.62 | +0.93% | 1.71 | 0.55 | 0.94 | -32.6% |

long_only's higher raw Sharpe is largely beta in costume: R2 0.32, and beta 0.35 x
the period's ~7-8%/y equity premium accounts for ~2.5-3%/y of its return.
long_short (beta 0.10, R2 0.03) is near-market-neutral — almost its entire return
is alpha. On alpha (the skill claim) long_short wins, +3.55 vs +2.81, with the
higher t. Neither alpha is statistically decisive (t < 2): long_short at 1.65 is
suggestive (p ~ 0.10), long_only at 1.41 is borderline-noise. t scales with
sqrt(T), so certifying long_short needs ~1.5x more sample (~another decade) — the
normal fate of a true ~0.4-Sharpe alpha stream, which is why breadth (many
signals) matters more than one signal clearing t=2.

**Stage 2 — long_short is a crisis-alpha strategy (the part the full-sample Sharpe
hid).** Regime CAGR / Sharpe:

| Regime | long_short | long_only |
|--------|-----------|-----------|
| GFC (07-10..09-03)        | +16.3% / 1.14 | -3.3% / -0.30 |
| QE calm (12-01..19-12)    | +6.0% / 0.56  | +7.2% / 0.64  |
| COVID (20-02..20-04)      | +33.8% / 1.59 | -45.3% / -1.18 |
| Inflation (21-04..23-07)  | +15.2% / 1.12 | +7.0% / 0.49  |

long_short made strong, high-Sharpe money in all three crises (both deflationary —
GFC, COVID — and inflationary); long_only was hurt in both deflationary ones. Best
regimes = market's worst regimes = crisis alpha, worth more than the 0.43
full-sample Sharpe implies. This confirms the design thesis argued during the
build: momentum is the general crash hedge (GFC/COVID, where InflationTrend was
quiet and the book went defensive via trend), and inflation protection is
regime-specific (2021-23, where the short-duration tilt paid). Both mechanisms
fired in their respective regimes.

**Inflation thesis vindicated.** The whole InflationTrend chain — keeping it
despite ~0 full-sample IC (regime-conditional), the shape-preserving /2 rescale
(not re-z-scoring), the no-demean net-short-duration tilt — was designed for the
inflation regime. Result there: +15.2% at Sharpe 1.12, MaxDD only -11.3%, while a
duration-heavy 60/40 had its worst year in decades. Survived contact with data.

**Drawdown anatomy.** long_short worst DD -21.5%: peak 2018-10-03 -> trough
2021-02-25 -> recovery 2022-03-07. Not the QE grind (that hypothesis was wrong) —
it's the directionless 2019 chop plus slow post-COVID recovery, and it resolves
when the inflation regime starts working (recovery = Mar-2022). long_short's pain
is boredom (no trends to trade); long_only's worst DD is the COVID crash itself
(-32.8%, pure beta). The two books' drawdowns have opposite character.

**Diversifier / portable-alpha test.** corr(long_short, 60/40) = +0.17. A 70/30
blend (70% 60/40 + 30% long_short) scores Sharpe 0.69 (beats 60/40's 0.62) with
MaxDD -19.4% (vs -32.6%). Higher Sharpe and shallower drawdown — the diversifier
signature. long_short was never competing with beta as a standalone
Sharpe-maximizer; it is additive to a beta portfolio, because its crisis-positive
returns offset 60/40's crisis losses. (70/30 is a round-number split, deliberately
not optimized — optimizing it would data-snoop the result.)

**Verdict.** long_short is a near-market-neutral crisis-alpha strategy whose value
is as a diversifier, not a standalone Sharpe play. long_only is a beta vehicle
with a side of alpha (most of its return is free via an index fund; no crisis
protection).

**Standing caveats (do not drop).** (1) n=1 inflation regime. (2) Same-rate
financing / no borrow fees flatters long_short specifically (its shorts). (3)
Every regime slice is low-N — behavioral illustrations, not significant sub-period
claims. (4) Neither full-sample alpha clears t=2.

## Phase 4 Results — Attribution (per-signal leave-one-in + per-bucket)

**Method.** scripts/evaluate_attribution.py, on the long_short book. Signals via
Approach A (leave-one-in): each run through the full stack alone, regime-sliced —
tests which signal provides which behavior, does not sum to combined. Buckets via
Approach B (held-weights x returns): sums exactly (ex cash leg). Plus a turnover
decomposition (leverage cap on vs off).

**Per-signal (Sharpe per regime) — the regime thesis, now measured not inferred.**

| | GFC | QE calm | COVID | Inflation | full |
|---|-----|---------|-------|-----------|------|
| TSMomentum     | +1.12 | +0.60 | +1.66 | +0.82 | +0.31 |
| BondCarry      | -1.19 | +0.33 | -1.68 | +0.45 | +0.16 |
| FXCarry        | -0.87 | -0.13 | +1.25 | +0.88 | +0.17 |
| InflationTrend | -0.74 | +0.29 | +0.55 | +1.29 | +0.28 |
| COMBINED       | +1.14 | +0.56 | +1.59 | +1.12 | +0.43 |

- Momentum owns the deflationary crashes (best Sharpe in GFC +1.12 and COVID
  +1.66) — trend flips defensive in any sustained selloff. The "momentum = general
  crash hedge" thesis confirmed at signal level.
- InflationTrend owns the inflation regime (+1.29, best there) and is negative in
  GFC — regime-specific, exactly as designed (the keep-despite-zero-IC, rescale,
  no-demean chain vindicated signal-by-signal).
- BondCarry is momentum's mirror: worst in both deflationary crises (-1.19, -1.68)
  because carry holds higher-yielding risk assets (HYG/EMB) crushed in
  flight-to-quality. A risk-on harvester — earns in calm, gives back in crashes.
  This offsetting crisis behavior vs momentum is why the blend smooths.
- The combined full-sample Sharpe (0.43) exceeds every individual signal (0.31,
  0.16, 0.17, 0.28): the diversification free lunch in one number. Four modest
  regime-specialized signals whose good/bad regimes don't coincide combine into
  something better than any alone — the cleanest evidence the 4-signal roster
  earns its complexity.

**Per-bucket contribution to combined (annualized, ex cash leg).**
credit +3.02% > equity +2.25% > fx +1.19% > real_assets +0.26% > rates -0.61% >
commodities -0.81%; TOTAL +5.31%/y (vs book CAGR 5.91%; the ~0.6% gap is the cash
leg on the ~0.54 uninvested fraction — reconciles, decomposition is exact).
- Prediction overturned: I expected rates/FX to dominate P&L (inverse-vol
  over-weights low-vol assets). Wrong — rates are the biggest position but a net
  P&L drag (-0.61%): a 40y bond-bull tail until 2022, then the inflation selloff
  where the short-duration tilt profited but held long-duration bled. Position
  size != P&L. Credit/equity are the actual engines (credit = where BondCarry, the
  strongest full-sample signal, lives).

**Turnover decomposition.** with-cap 10.12x/y, cap-OFF 12.06x/y, so the leverage
cap reduces turnover by 1.94x/y (it clamps high-gross days, absorbing gross
volatility rather than chasing it).
- Prediction overturned: the deferred worry was cap-toggling inflating turnover.
  Opposite. Deferred item downgraded: "smooth k_t to cut turnover" is now
  not-worth-it — the cap suppresses turnover for free, and residual 10.1x/y at 1bp
  is ~10bps/y, trivial. (Limit: this separates cap-toggling cleanly but doesn't
  split the 12.06 into signal vs vol-scalar — entangled; not worth building given
  turnover is a non-issue cost-wise.)

**Net.** The crisis-alpha verdict is fully attributed: momentum drives
deflationary-crisis protection, inflation drives inflation-regime protection,
carry harvests calm and provides the full-sample base, and the four beat any one
alone. Every regime-table inference is now a measurement.

## Phase 4 Results — Sensitivity sweeps (one-at-a-time)

**Method.** scripts/evaluate_sensitivity.py, long_short. Each parameter swept with
the other three at baseline (vol_window 126, rebalance ME, max_gross 3.0, cost
1bp). No "best" setting picked — the test is whether the baseline verdict
(crisis-alpha + diversifier + inflation protection) is typical among neighbors or
an outlier. Tracks Sharpe, alpha, alpha-t, maxDD, the 70/30-blend Sharpe
(diversifier claim) and inflation-regime Sharpe (protection claim), so the verdict
is stressed, not just the headline. Same 22y sample: tests fragility-to-knobs, not
out-of-sample (walk-forward is a separate exercise).

**Result: robust on 3 of 4; baseline is typical (not a peak), no overfit
signature.**

- Vol window {63,126,252}: Sharpe 0.42/0.43/0.43, alpha +3.42/+3.55/+3.68%, infl
  Sh +1.01/+1.12/+1.20. Flat, with a mild monotonic improvement toward longer
  windows (slow signals), within noise. 126 is not a lucky pick, and I'm not
  switching to 252 on a 0.15 infl-slice difference (that would data-snoop).
- Leverage cap {2.0,3.0,4.0}: Sharpe 0.40/0.43/0.41, maxDD -18.4/-21.5/-24.9%. The
  maxDD drift is the cap working (tighter = shallower DD and slightly less return;
  looser = deeper DD as the levered tail passes) — a risk/return tradeoff, not
  fragility. 3.0 sits sensibly mid.
- Cost {0.5,1,2,5 bps}: alpha +3.60/+3.55/+3.44/+3.14%, erodes gracefully. Even at
  5bps (5x baseline, punitive for ETFs) alpha stays +3.14% and the blend 0.68 —
  the edge is not a low-cost artifact, it survives pessimistic execution. (The key
  test given ~10x/y turnover.)
- Diversifier blend (0.68-0.72) and inflation Sharpe (+1.0-1.2) are rock-stable
  across all the above — the two verdict claims, not just headline Sharpe, are
  robust to parameter choice.

**The one sensitivity: rebalance frequency is a threshold, not a smooth tradeoff.**
W 0.41 / ME 0.43 / QE 0.10 (alpha -0.84%, t=-0.34, maxDD -45.99%). Quarterly
destroys the strategy. The mechanism is interpretable and validates the design: at
~63-day cadence the book drifts untouched too long — the vol-target scalar can't
de-lever into a vol spike (it rides it at stale too-high leverage, hence the -46%
DD) and signals go stale a quarter past when they should flip. Monthly is frequent
enough to react; weekly ties monthly (0.41) but adds cost for no benefit.

Conclusion: monthly is not arbitrary — it's near the slow edge of the viable
range. There's a floor between monthly and quarterly below which the risk
machinery can't react. Documented operational constraint: monthly-or-faster
rebalancing is required; monthly is the lowest-cost viable cadence. (Mild
robustness caveat: a strategy viable at any cadence would be more robust; the
cliff is a dependency to state plainly, not a defect, since monthly is well inside
the safe zone.)

**Net.** The crisis-alpha / diversifier / inflation-protection verdict is robust
to the 3 parameters that could expose overfitting; the baseline is typical in all
3. The single dependency (monthly-or-faster rebalancing) has a clean economic
mechanism. Same-sample robustness only — not a substitute for out-of-sample
validation. Standing caveats unchanged (n=1 inflation, same-rate financing
flatters long_short).
