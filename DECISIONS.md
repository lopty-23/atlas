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