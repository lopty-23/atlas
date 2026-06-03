"""Information Coefficient (IC): how well a signal predicts forward returns.

This module is signal-agnostic: it works on any date x asset score frame,
which is what every Signal.compute() returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Minimum assets with both a signal and a return on a date, for that date's
# cross-sectional correlation to be meaningful. Below this, the date is skipped.
MIN_ASSETS_PER_DATE = 5


def compute_ic_timeseries(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    method: str = "spearman",
    min_assets: int = MIN_ASSETS_PER_DATE,
    lag: int = 1,
) -> pd.Series:
    """Per-date cross-sectional IC between a (lagged) signal and forward returns.

    On each date, compute the rank correlation across assets between the
    signal scores and the forward returns. Returns one IC value per date
    (NaN on dates with fewer than `min_assets` valid pairs).
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
    if lag < 0:
        raise ValueError(f"lag must be non-negative, got {lag}")

    lagged_signal = signal.shift(lag)

    common_cols = lagged_signal.columns.intersection(fwd_returns.columns)
    sig = lagged_signal[common_cols]
    ret = fwd_returns.reindex(index=lagged_signal.index, columns=common_cols)

    ics: dict[pd.Timestamp, float] = {}
    for date in sig.index:
        s_row = sig.loc[date]
        r_row = ret.loc[date]
        valid = s_row.notna() & r_row.notna()
        if valid.sum() < min_assets:
            ics[date] = np.nan
            continue
        s_valid = s_row[valid]
        r_valid = r_row[valid]
        if s_valid.nunique() < 2 or r_valid.nunique() < 2:
            ics[date] = np.nan
            continue
        ic = s_valid.corr(r_valid, method=method)
        ics[date] = ic

    return pd.Series(ics, name="ic")


def compute_ic(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    method: str = "spearman",
    min_assets: int = MIN_ASSETS_PER_DATE,
    lag: int = 1,
) -> float:
    ic_ts = compute_ic_timeseries(
        signal, fwd_returns, method=method, min_assets=min_assets, lag=lag
    )
    return ic_ts.mean()