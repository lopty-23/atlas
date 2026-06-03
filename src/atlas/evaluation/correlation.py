"""Signal correlation: how independent are the signals from each other?

Two distinct questions:

1. signal_value_correlation — "do they POSITION similarly?" Correlates the raw
   score frames. Redundant positioning means combining two signals just doubles
   one bet. Undefined for signals covering disjoint assets (which is itself
   informative: disjoint signals automatically diversify in positioning).

2. ic_correlation — "do they WORK at the same times?" Correlates each signal's
   per-date IC series. Two signals can hold different assets yet earn in the
   same regimes (correlated returns); this matrix reveals that. It is the key
   input to the keep/drop and blend-weight decisions: a weak standalone signal
   whose IC is UNCORRELATED with the strong signals can still add ballast.
"""
from __future__ import annotations

import pandas as pd

from atlas.data.returns import forward_returns
from atlas.evaluation.ic import compute_ic_timeseries, MIN_ASSETS_PER_DATE


def signal_value_correlation(
    signals: dict[str, pd.DataFrame],
    method: str = "spearman",
    min_shared_assets: int = 3,
) -> pd.DataFrame:
    """Average cross-sectional positioning similarity between signal pairs.

    For each pair, on each date compute the cross-sectional correlation between
    their scores over the assets BOTH cover, then average over dates. Returns a
    signals x signals matrix. NaN where two signals never share enough assets.
    """
    names = list(signals)
    out = pd.DataFrame(index=names, columns=names, dtype=float)

    for a in names:
        for b in names:
            if a == b:
                out.loc[a, b] = 1.0
                continue
            sa, sb = signals[a], signals[b]
            shared_cols = sa.columns.intersection(sb.columns)
            if len(shared_cols) < min_shared_assets:
                out.loc[a, b] = float("nan")
                continue
            # Align on shared assets and common dates.
            fa = sa[shared_cols]
            fb = sb[shared_cols].reindex(index=fa.index)

            per_date = []
            for date in fa.index:
                ra, rb = fa.loc[date], fb.loc[date]
                valid = ra.notna() & rb.notna()
                if valid.sum() < min_shared_assets:
                    continue
                rav, rbv = ra[valid], rb[valid]
                if rav.nunique() < 2 or rbv.nunique() < 2:
                    continue
                per_date.append(rav.corr(rbv, method=method))
            out.loc[a, b] = float(pd.Series(per_date).mean()) if per_date else float("nan")

    return out


def ic_correlation(
    signals: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    horizon: int,
    min_assets_map: dict[str, int] | None = None,
    method: str = "spearman",
    lag: int = 1,
) -> pd.DataFrame:
    """Correlation of the signals' per-date IC series at a given horizon.

    Builds each signal's IC time series (its edge over time) at `horizon`, then
    correlates those series pairwise. High positive correlation = the two
    signals' edges show up together (little return diversification); near-zero
    or negative = they earn at different times (genuine diversification).
    """
    fwd = forward_returns(prices, horizon=horizon)
    min_assets_map = min_assets_map or {}

    # Build each signal's IC time series.
    ic_series: dict[str, pd.Series] = {}
    for name, sig in signals.items():
        ma = min_assets_map.get(name, MIN_ASSETS_PER_DATE)
        ic_series[name] = compute_ic_timeseries(
            sig, fwd, method=method, min_assets=ma, lag=lag
        )

    # Assemble into one frame and correlate (pairwise, NaN-aware).
    ic_frame = pd.DataFrame(ic_series)
    return ic_frame.corr(method=method, min_periods=30)