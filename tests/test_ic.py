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

from atlas.evaluation.ic import (
    compute_ic,
    compute_ic_decay,
    compute_ic_timeseries,
    compute_icir,
    compute_rolling_ic,
)

def _random_prices(n_dates: int = 400, n_assets: int = 20, seed: int = 0) -> pd.DataFrame:
    """Random-walk prices (cumulative product of small random returns)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_dates)
    cols = [f"A{i}" for i in range(n_assets)]
    rets = rng.normal(0, 0.01, size=(n_dates, n_assets))
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return pd.DataFrame(prices, index=idx, columns=cols)


def _trending_prices(n_dates: int = 400, n_assets: int = 20, seed: int = 0) -> pd.DataFrame:
    """Prices with persistent per-asset drift, so momentum is predictive.

    Each asset gets a fixed random drift; the ones drifting up keep drifting up,
    so trailing momentum predicts forward returns (a clean trend to detect)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_dates)
    cols = [f"A{i}" for i in range(n_assets)]
    drifts = rng.normal(0, 0.0008, size=n_assets)  # persistent per-asset drift
    noise = rng.normal(0, 0.005, size=(n_dates, n_assets))
    rets = drifts[None, :] + noise
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return pd.DataFrame(prices, index=idx, columns=cols)

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


class TestICIR:
    """Rolling IC and the HAC-corrected ICIR / t-stat."""

    def test_rolling_ic_smooths_noise(self) -> None:
        ret = _random_returns(n_dates=600)
        signal = ret.shift(-1)  # near-perfect predictor (lead-shifted for the lag)
        rolling = compute_rolling_ic(signal, ret, window=100)
        # A strong predictor's rolling IC should be high and stable once warm.
        warm = rolling.dropna()
        assert warm.min() > 0.9, "Rolling IC of a strong signal should stay high"

    def test_icir_positive_for_real_signal(self) -> None:
        ret = _random_returns(n_dates=600, seed=4)
        rng = np.random.default_rng(11)
        noise = pd.DataFrame(
            rng.normal(0, 0.01, size=ret.shape), index=ret.index, columns=ret.columns
        )
        signal = (ret + noise).shift(-1)
        result = compute_icir(signal, ret, horizon=1)
        assert result["mean_ic"] > 0
        assert result["icir"] > 0
        assert result["t_stat"] > 2, "A real signal should be statistically significant"

    def test_icir_insignificant_for_noise(self) -> None:
        ret = _random_returns(n_dates=600, seed=5)
        rng = np.random.default_rng(22)
        signal = pd.DataFrame(
            rng.normal(0, 1, size=ret.shape), index=ret.index, columns=ret.columns
        )
        result = compute_icir(signal, ret, horizon=1)
        # Noise should not be significant.
        assert abs(result["t_stat"]) < 2, "Noise should not be statistically significant"

    def test_hac_se_exceeds_naive_under_overlap(self) -> None:
        """With autocorrelated (overlapping) IC, the HAC SE should exceed the
        naive SE — the whole point of the correction."""
        # Build an autocorrelated IC series directly and compare SEs.
        from atlas.evaluation.ic import _newey_west_se
        rng = np.random.default_rng(33)
        # AR(1) series: each value correlated with the previous -> overlap-like.
        n = 500
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.7 * x[i - 1] + rng.normal(0, 1)
        s = pd.Series(x)
        naive_se = s.std() / np.sqrt(len(s))
        hac_se = _newey_west_se(s, n_lags=10)
        assert hac_se > naive_se, (
            "Under positive autocorrelation, HAC SE must exceed the naive SE"
        )

class TestICDecay:
    """The decay curve: IC across multiple forward horizons."""

    def test_decay_has_row_per_horizon(self) -> None:
        prices = _random_prices()
        signal = pd.DataFrame(
            np.random.default_rng(0).normal(0, 1, size=prices.shape),
            index=prices.index, columns=prices.columns,
        )
        decay = compute_ic_decay(signal, prices, horizons=(1, 5, 21))
        assert list(decay.index) == [1, 5, 21]
        assert "mean_ic" in decay.columns

    def test_longer_horizon_peak_for_persistent_signal(self) -> None:
        """A signal built to predict the LONGER-horizon return should show its
        strongest IC at the longer horizon, not the 1-day."""
        prices = _trending_prices()
        # Signal = trailing momentum (price now vs 60 days ago). For a trending
        # series this predicts longer-horizon forward returns better than 1-day.
        from atlas.data.returns import trailing_returns
        signal = trailing_returns(prices, lookback=60)
        decay = compute_ic_decay(
            signal, prices, horizons=(1, 63), include_icir=False
        )
        assert decay.loc[63, "mean_ic"] > decay.loc[1, "mean_ic"], (
            "A momentum signal on trending prices should predict the 63-day "
            "return better than the 1-day return"
        )

    def test_icir_columns_present_when_requested(self) -> None:
        prices = _random_prices()
        signal = pd.DataFrame(
            np.random.default_rng(1).normal(0, 1, size=prices.shape),
            index=prices.index, columns=prices.columns,
        )
        decay = compute_ic_decay(signal, prices, horizons=(1, 21), include_icir=True)
        assert {"mean_ic", "icir", "t_stat", "n_obs"} <= set(decay.columns)