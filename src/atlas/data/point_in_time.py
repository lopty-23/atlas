"""Apply publication lags and vintage tracking to FRED data.

This module handles two types of correction:

1. FIXED LAG (`apply_publication_lags`): For market-based series where
   revisions don't materially happen (yields, FX, VIX, breakevens),
   shift the reference date forward by a fixed publication lag.

2. VINTAGES (`align_vintages_to_calendar`): For revision-heavy series
   (GDP, payrolls, CPI), use the long-format vintage data to know
   exactly what was publicly known on each date — including which
   revision was the latest at that moment.

Both produce a DataFrame indexed by trading-day calendar, where every 
cell details what is publicly known on that day
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def lag_map_from_config(universe_config: dict[str, Any]) -> dict[str, int]:
    """Extract a {series_id: lag_days} dict for NON-VINTAGE series only.

    Vintage-aware series are excluded — they use real publication dates from
    ALFRED and don't need fixed lags.
    """
    return {
        item["id"]: item["lag_days"]
        for group in universe_config["fred_series"].values()
        for item in group
        if not item.get("vintage", False)
    }


def vintage_series_from_config(universe_config: dict[str, Any]) -> list[str]:
    """Return the list of series IDs flagged for vintage-aware handling."""
    return [
        item["id"]
        for group in universe_config["fred_series"].values()
        for item in group
        if item.get("vintage", False)
    ]


def apply_publication_lags(
    raw_fred: pd.DataFrame,
    lag_map: dict[str, int],
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Fixed lag"""
    out = pd.DataFrame(index=target_index)

    for series_id, lag in lag_map.items():
        if series_id not in raw_fred.columns:
            continue

        ref_series = raw_fred[series_id].dropna()
        if ref_series.empty:
            continue

        # Shift reference dates forward by `lag` business days.
        pub_dates = ref_series.index + pd.tseries.offsets.BDay(lag)
        pub_series = pd.Series(ref_series.values, index=pub_dates, name=series_id)

        # Handle duplicate publication dates by keeping the latest value.
        pub_series = pub_series[~pub_series.index.duplicated(keep="last")]

        # Reindex onto target business-day calendar with forward-fill.
        aligned = pub_series.reindex(target_index, method="ffill")
        out[series_id] = aligned

    return out


def align_vintages_to_calendar(
    vintages: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Keep only the latest observation from each vintage, then forward-fill onto the calendar."""
    out = pd.DataFrame(index=target_index)

    for series_id in vintages["series_id"].unique():
        sub = vintages[vintages["series_id"] == series_id].copy()

        latest_per_vintage = sub.groupby("vintage_date", as_index=True).last()

        pub_series = pd.Series(
            latest_per_vintage["value"].values,
            index=pd.DatetimeIndex(latest_per_vintage.index),
            name=series_id,
        )
        pub_series = pub_series.sort_index()

        aligned = pub_series.reindex(target_index, method="ffill")
        out[series_id] = aligned

    return out

# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def build_pit_macro(
    raw_fred: pd.DataFrame,
    vintages: pd.DataFrame,
    prices: pd.DataFrame,
    universe_config: dict[str, Any],
) -> pd.DataFrame:
    """Build the unified point-in-time macro DataFrame including both 
    fixed-lag and vintage paths"""

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices.index must be a DatetimeIndex")

    target_index = prices.index

    # Fixed-lag path
    lag_map = lag_map_from_config(universe_config)
    lagged = apply_publication_lags(raw_fred, lag_map, target_index)

    # Vintage path
    aligned_vintages = align_vintages_to_calendar(vintages, target_index)

    # Combine. Vintage series take precedence if a series ID appears in both
    # (it shouldn't, but be defensive).
    combined = lagged.join(aligned_vintages, how="outer", rsuffix="_v")
    duplicate_cols = [c for c in combined.columns if c.endswith("_v")]
    if duplicate_cols:
        raise ValueError(
            f"Series IDs appear in both vintage and non-vintage paths: {duplicate_cols}"
        )

    return combined