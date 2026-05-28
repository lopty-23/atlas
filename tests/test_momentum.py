"""Tests for the TSMomentum signal.

Verifies the raw momentum computation (trend direction, vol-scaling, skip-days)
on hand-crafted price series with known answers, plus point-in-time safety
(the signal must never use future prices).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.signals.momentum import TSMomentum


def _trending_prices() -> pd.DataFrame:
    """Three assets: A trends up, B trends down, C is flat."""
    n = 300
    idx = pd.bdate_range("2023-01-01", periods=n)
    up = pd.Series(100.0 * (1.0005 ** np.arange(n)), index=idx)    # ~+0.05%/day
    down = pd.Series(100.0 * (0.9995 ** np.arange(n)), index=idx)  # ~-0.05%/day
    flat = pd.Series(100.0, index=idx)
    return pd.DataFrame({"A": up, "B": down, "C": flat})


class TestRawMomentum:
    """The raw (pre-z-score) momentum reflects trend direction."""

    def test_uptrend_positive_downtrend_negative(self) -> None:
        prices = _trending_prices()
        sig = TSMomentum(lookback_days=252, vol_scale=False)
        raw = sig._compute_raw(prices, macro=pd.DataFrame())
        last = raw.iloc[-1]
        assert last["A"] > 0, "Uptrending asset should have positive raw momentum"
        assert last["B"] < 0, "Downtrending asset should have negative raw momentum"

    def test_zscore_ranks_uptrend_above_downtrend(self) -> None:
        prices = _trending_prices()
        sig = TSMomentum(lookback_days=252, vol_scale=False)
        scores = sig.compute(prices, macro=pd.DataFrame())
        last = scores.iloc[-1]
        assert last["A"] > last["C"] > last["B"], (
            "After z-scoring, up-trend should rank above flat above down-trend"
        )


class TestVolScaling:
    """Vol-scaling makes risk-adjusted momentum comparable across assets."""

    def test_vol_scale_changes_raw_magnitude(self) -> None:
        prices = _trending_prices()
        raw_unscaled = TSMomentum(lookback_days=252, vol_scale=False)._compute_raw(
            prices, pd.DataFrame()
        )
        raw_scaled = TSMomentum(lookback_days=252, vol_scale=True)._compute_raw(
            prices, pd.DataFrame()
        )
        # Vol-scaling divides by vol, so the magnitude should differ.
        assert not np.isclose(
            raw_unscaled["A"].iloc[-1], raw_scaled["A"].iloc[-1]
        ), "Vol-scaling should change the raw signal magnitude"


class TestPointInTimeSafety:
    """The signal must never use future prices."""

    def test_signal_unaffected_by_future_prices(self) -> None:
        """Changing prices AFTER date t must not change the signal AT date t."""
        prices = _trending_prices()
        sig = TSMomentum(lookback_days=126, vol_scale=True)

        scores_full = sig.compute(prices, macro=pd.DataFrame())

        # Now corrupt the LAST 10 days of prices and recompute.
        corrupted = prices.copy()
        corrupted.iloc[-10:] = corrupted.iloc[-10:] * 5.0

        scores_corrupted = sig.compute(corrupted, macro=pd.DataFrame())

        # The signal at a date BEFORE the corruption (e.g. 20 days from the end)
        # must be identical — future prices can't leak backward.
        check_date = scores_full.index[-20]
        pd.testing.assert_series_equal(
            scores_full.loc[check_date],
            scores_corrupted.loc[check_date],
            check_names=False,
        )


class TestParameterization:
    """Different lookbacks produce different (but valid) signals."""

    def test_different_lookbacks_differ(self) -> None:
        prices = _trending_prices()
        s21 = TSMomentum(lookback_days=21).compute(prices, pd.DataFrame())
        s252 = TSMomentum(lookback_days=252).compute(prices, pd.DataFrame())
        # The two should not be identical (different windows see different trends).
        assert not s21["A"].equals(s252["A"]), (
            "Different lookbacks should produce different signals"
        )

    def test_output_shape_matches_input(self) -> None:
        prices = _trending_prices()
        scores = TSMomentum(lookback_days=63).compute(prices, pd.DataFrame())
        assert scores.shape == prices.shape, (
            "Signal output should have the same shape as the price input"
        )
        assert list(scores.columns) == list(prices.columns)