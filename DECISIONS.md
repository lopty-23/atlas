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