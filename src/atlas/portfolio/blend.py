"""Signal blending: combine per-asset signal scores into one composite.

The blend is a per-date, per-asset weighted average over the signals present
(non-NaN) at that cell, with weights renormalized over the present set. Point-in-time safe: 
a pure cross-sectional operation at each date, no time-axis leakage. Does not apply
the execution lag as that would be applied in the backtest (weights_today =
f(signals_yesterday)).
"""

from __future__ import annotations

import pandas as pd


def blend_signals(
    signals: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
) -> pd.DataFrame:
    names = list(signals)
    if weights is None:
        weights = {name: 1.0 / len(names) for name in names}
    if scales is None:
        scales = {}

    # Union grid: every date and asset appearing in any signal (missing -> NaN).
    all_dates = signals[names[0]].index
    all_assets = signals[names[0]].columns
    for name in names[1:]:
        all_dates = all_dates.union(signals[name].index)
        all_assets = all_assets.union(signals[name].columns)

    weighted_sum = pd.DataFrame(0.0, index=all_dates, columns=all_assets)
    present_weight = pd.DataFrame(0.0, index=all_dates, columns=all_assets)

    for name in names:
        w = weights[name]
        scaled = signals[name].reindex(index=all_dates, columns=all_assets) / scales.get(name, 1.0)
        present = scaled.notna()
        # A NaN cell adds 0 to the numerator AND 0 to the present weight, so it is
        # excluded from the average -- not silently treated as a zero score.
        weighted_sum = weighted_sum + (scaled * w).fillna(0.0)
        present_weight = present_weight + present * w

    # Divide by PRESENT weight (the renormalization); all-absent cells -> NaN.
    denom = present_weight.where(present_weight > 0)
    return weighted_sum.div(denom)