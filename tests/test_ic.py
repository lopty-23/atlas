"""Tests for the IC framework.

The core tests use SYNTHETIC signals with a KNOWN relationship to returns:
- a signal equal to the forward return -> IC should be ~1
- a signal independent of returns -> IC should be ~0
- a signal that is the return reversed -> IC should be ~-1
If compute_ic recovers these, the implementation is correct.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.evaluation.ic import compute_ic, compute_ic_timeseries


def _random_returns(n_dates: int = 300, n_assets: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_dates)
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(rng.normal(0, 0.01, size=(n_dates, n_assets)), index=idx, columns=cols)


class TestKnownIC:
    """Synthetic signals with a known relationship to returns.

    Because compute_ic lags the signal by 1 (execution delay) before
    correlating, we construct each test signal to LEAD the return by one day
    (signal = relationship.shift(-1)), so that after the internal lag it aligns
    with the return it's meant to predict.
    """

    def test_perfect_signal_ic_near_one(self) -> None:
        ret = _random_returns()
        # Signal leads the return by one day -> after the internal lag, the
        # signal aligns with the return it predicts -> IC ~ 1.
        signal = ret.shift(-1)
        ic = compute_ic(signal, ret)
        assert ic > 0.99, f"A perfectly-predictive signal should have IC ~1, got {ic:.3f}"

    def test_inverted_signal_ic_near_minus_one(self) -> None:
        ret = _random_returns()
        signal = -ret.shift(-1)
        ic = compute_ic(signal, ret)
        assert ic < -0.99, f"A perfectly-inverted signal should have IC ~-1, got {ic:.3f}"

    def test_noise_signal_ic_near_zero(self) -> None:
        ret = _random_returns(seed=1)
        rng = np.random.default_rng(999)
        signal = pd.DataFrame(
            rng.normal(0, 1, size=ret.shape), index=ret.index, columns=ret.columns
        )
        ic = compute_ic(signal, ret)
        assert abs(ic) < 0.05, f"An independent-noise signal should have IC ~0, got {ic:.3f}"

    def test_partial_signal_recovers_expected_ic(self) -> None:
        ret = _random_returns(seed=2)
        rng = np.random.default_rng(7)
        noise = pd.DataFrame(
            rng.normal(0, 0.01, size=ret.shape), index=ret.index, columns=ret.columns
        )
        signal = (ret + noise).shift(-1)  # leads by one day, then internal lag aligns it
        ic = compute_ic(signal, ret)
        assert 0.3 < ic < 0.9, (
            f"A return+equal-noise signal should have a moderate positive IC, got {ic:.3f}"
        )

    def test_lag_zero_recovers_contemporaneous(self) -> None:
        """With lag=0, a signal equal to the (contemporaneous) return gives IC ~1.
        This confirms the lag parameter actually shifts the alignment."""
        ret = _random_returns(seed=3)
        signal = ret.copy()  # no lead; lag=0 means no internal shift
        ic = compute_ic(signal, ret, lag=0)
        assert ic > 0.99, f"With lag=0, signal==return should give IC ~1, got {ic:.3f}"


class TestThinCrossSection:
    """Dates with too few assets are skipped."""

    def test_few_assets_skipped(self) -> None:
        ret = _random_returns(n_assets=3)  # only 3 assets, below the min of 5
        signal = ret.copy()
        ic_ts = compute_ic_timeseries(signal, ret, min_assets=5)
        assert ic_ts.isna().all(), "With fewer than min_assets, all dates should be NaN"

    def test_partial_coverage_uses_valid_assets_only(self) -> None:
        ret = _random_returns(n_assets=20)
        signal = ret.shift(-1)  # lead by one day so the internal lag aligns it
        # Blank the signal on all but 6 assets (simulating a signal that abstains).
        signal.iloc[:, 6:] = np.nan
        ic = compute_ic(signal, ret, min_assets=5)
        # Still computable on the 6 covered assets; should recover ~1 (signal=ret).
        assert ic > 0.99, "IC should compute over the covered assets and ignore NaN ones"


class TestMethodValidation:
    def test_invalid_method_raises(self) -> None:
        ret = _random_returns()
        import pytest
        with pytest.raises(ValueError, match="spearman"):
            compute_ic(ret, ret, method="kendall")