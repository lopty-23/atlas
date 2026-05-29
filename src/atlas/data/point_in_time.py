"""Apply publication lags and vintage tracking to FRED data.

Every macro release has a publication delay between the reference date
(the period the data describes) and the publication date (when it became
publicly known). A backtest using values between those two dates is
cheating — it uses information that didn't yet exist.

This module handles two flavors of correction:

1. FIXED LAG (`apply_publication_lags`): For market-based series where
   revisions don't materially happen (yields, FX, VIX, breakevens),
   shift the reference date forward by a fixed publication lag.

2. VINTAGES (`align_vintages_to_calendar`): For revision-heavy series
   (GDP, payrolls, CPI), use the long-format vintage data to know
   exactly what was publicly known on each date — including which
   *revision* was the latest at that moment.

The two paths converge: both produce a DataFrame indexed by trading-day
calendar, where every cell is "what was publicly known on date t."

Public entry points:
    apply_publication_lags(raw_fred, lag_map, target_index) -> DataFrame
    align_vintages_to_calendar(vintages, target_index) -> DataFrame
    build_pit_macro(raw_fred, vintages, prices, universe) -> DataFrame
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
    """Shift each FRED series forward by its publication lag, then align onto
    a daily business-day index using forward-fill.

    Use this for market-based series only. For revision-heavy series, use
    `align_vintages_to_calendar` instead."""
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

        # Handle (rare) duplicate publication dates by keeping the latest value.
        pub_series = pub_series[~pub_series.index.duplicated(keep="last")]

        # Reindex onto target business-day calendar with forward-fill.
        aligned = pub_series.reindex(target_index, method="ffill")
        out[series_id] = aligned

    return out


def align_vintages_to_calendar(
    vintages: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Convert long-format vintage data into a daily point-in-time wide frame.

    Each cell (t, series) is the value of `series` for the LATEST observation
    period available in the LATEST vintage published on or before t. This is
    "what was knowable about this series at close of business on date t,"
    fully respecting revisions.

    The vintage data is long-format: for each vintage_date (a publication
    event), it contains MANY rows — one per observation_date (every period
    that publication reported, often going back decades). For each vintage we
    must take the value for its LATEST observation_date (the freshest period
    that publication covered), NOT an arbitrary historical observation.
    """
    out = pd.DataFrame(index=target_index)

    for series_id in vintages["series_id"].unique():
        sub = vintages[vintages["series_id"] == series_id].copy()

        # For each vintage (publication event), keep only the row for the
        # LATEST observation period that vintage reported. Sorting by both
        # keys then taking the last row per vintage_date guarantees we select
        # the newest observation within each vintage, not an arbitrary one.
        sub = sub.sort_values(["vintage_date", "observation_date"])
        latest_per_vintage = sub.groupby("vintage_date", as_index=True).last()

        # latest_per_vintage is now indexed by vintage_date, one row each,
        # holding that vintage's most-recent-period value.
        pub_series = pd.Series(
            latest_per_vintage["value"].values,
            index=pd.DatetimeIndex(latest_per_vintage.index),
            name=series_id,
        )
        pub_series = pub_series.sort_index()

        # Forward-fill onto the trading calendar: on date t, the value is the
        # latest vintage published on or before t.
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
    """Build the unified point-in-time macro DataFrame.

    Combines the fixed-lag path (for market series) and the vintage path
    (for revision-heavy series) into one wide DataFrame indexed by the
    prices' trading calendar."""
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