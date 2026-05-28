"""Tests for the Signal abstract base class.

Verifies the contract (can't instantiate abstract, subclasses must implement
_compute_raw) and the post-processing pipeline (winsorization + cross-sectional
z-scoring produce mean~0, std~1 per row).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas.signals.base import Signal


class _DummySignal(Signal):
    """Minimal concrete signal for testing: raw signal = trailing 1d return."""

    def _compute_raw(self, prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
        return prices.pct_change()


class _IdentitySignal(Signal):
    """Returns the prices unchanged as the raw signal (for controlled tests)."""

    def _compute_raw(self, prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
        return prices.copy()


class TestAbstractContract:
    """The base class enforces the implementation contract."""

    def test_cannot_instantiate_base_class(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            Signal()  # type: ignore[abstract]

    def test_subclass_with_compute_raw_works(self) -> None:
        sig = _DummySignal()
        assert sig.name == "_DummySignal"

    def test_custom_name(self) -> None:
        sig = _DummySignal(name="my_momentum")
        assert sig.name == "my_momentum"


class TestZScoring:
    """Cross-sectional z-scoring produces mean~0, std~1 per row."""

    def test_each_row_has_zero_mean(self) -> None:
        prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, 102.0, 103.0],
                "B": [100.0, 99.0, 98.0, 97.0],
                "C": [100.0, 100.0, 101.0, 102.0],
            },
            index=pd.bdate_range("2024-01-01", periods=4),
        )
        scores = _DummySignal().compute(prices, macro=pd.DataFrame())
        # Skip the first row (all NaN from pct_change); the rest should have
        # cross-sectional mean ~ 0.
        row_means = scores.iloc[1:].mean(axis=1)
        assert np.allclose(row_means.values, 0.0, atol=1e-9), (
            "Each row should have zero cross-sectional mean after z-scoring"
        )

    def test_each_row_has_unit_std(self) -> None:
        prices = pd.DataFrame(
            {
                "A": [100.0, 101.0, 103.0, 106.0],
                "B": [100.0, 99.0, 97.0, 94.0],
                "C": [100.0, 100.5, 101.0, 101.5],
            },
            index=pd.bdate_range("2024-01-01", periods=4),
        )
        scores = _DummySignal().compute(prices, macro=pd.DataFrame())
        row_stds = scores.iloc[1:].std(axis=1)
        assert np.allclose(row_stds.values, 1.0, atol=1e-9), (
            "Each row should have unit cross-sectional std after z-scoring"
        )

    def test_level_invariance(self) -> None:
        """A signal sitting at a high level vs a low level should produce the
        SAME z-scores if the relative pattern across assets is identical.

        This is the property the user asked about: absolute level is removed
        by mean-subtraction; only the cross-sectional pattern survives.
        """
        # Two raw signals with identical SHAPE but different LEVELS.
        low_level = pd.DataFrame(
            {"A": [0.12], "B": [0.18], "C": [0.15]},
            index=pd.bdate_range("2024-01-01", periods=1),
        )
        high_level = low_level + 0.7  # shift everything up by 0.7

        z_low = Signal._cross_sectional_zscore(low_level)
        z_high = Signal._cross_sectional_zscore(high_level)

        assert np.allclose(z_low.values, z_high.values), (
            "Z-scores should be identical regardless of absolute level — "
            "mean-subtraction removes the level"
        )

    def test_higher_raw_gets_higher_zscore(self) -> None:
        """Within a row, a higher raw value should map to a higher z-score."""
        raw = pd.DataFrame(
            {"A": [1.0], "B": [2.0], "C": [3.0]},
            index=pd.bdate_range("2024-01-01", periods=1),
        )
        z = Signal._cross_sectional_zscore(raw)
        assert z["A"].iloc[0] < z["B"].iloc[0] < z["C"].iloc[0], (
            "Z-score ordering should preserve raw value ordering"
        )


class TestWinsorization:
    """Winsorization clips outliers before z-scoring."""

    def test_outlier_is_clipped(self) -> None:
        """An extreme outlier should be pulled in toward the robust band."""
        # 9 normal values around 0, one huge outlier.
        row = {f"asset_{i}": [float(i)] for i in range(9)}
        row["outlier"] = [1000.0]
        df = pd.DataFrame(row, index=pd.bdate_range("2024-01-01", periods=1))

        clipped = Signal._winsorize(df, limit=4.0)
        # The outlier should now be drastically smaller than 1000.
        assert clipped["outlier"].iloc[0] < 1000.0, "Outlier should be clipped down"

        # It should sit at the robust upper band: median + 4 * (1.4826 * MAD).
        row_median = df.median(axis=1).iloc[0]
        mad = df.sub(df.median(axis=1), axis=0).abs().median(axis=1).iloc[0]
        expected_upper = row_median + 4.0 * (1.4826 * mad)
        assert np.isclose(clipped["outlier"].iloc[0], expected_upper), (
            "Clipped outlier should sit at the robust upper band (median + 4*1.4826*MAD)"
        )

    def test_normal_values_untouched(self) -> None:
        """Values within ±limit std should pass through unchanged."""
        df = pd.DataFrame(
            {"A": [1.0], "B": [2.0], "C": [3.0], "D": [4.0]},
            index=pd.bdate_range("2024-01-01", periods=1),
        )
        clipped = Signal._winsorize(df, limit=4.0)
        # None of these are beyond 4 std, so all should be unchanged.
        assert np.allclose(clipped.values, df.values), (
            "Values within the band should not be modified"
        )