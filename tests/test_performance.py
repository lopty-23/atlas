"""Tests for performance metrics. Hand-computed known answers on tiny series."""
from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.backtest.performance import (
    cagr, annualized_vol, sharpe, sortino, max_drawdown, calmar,
    time_underwater, hit_rate, summarize,
)

IDX4 = pd.RangeIndex(4)
ZERO_RF = pd.Series(0.0, index=IDX4)


class TestCagr:
    def test_known_answer(self):
        # Equity = 1.1 * 0.9 * 1.05 * 1.0 = 1.0395 over 4/252 years.
        r = pd.Series([0.10, -0.10, 0.05, 0.0], index=IDX4)
        assert np.isclose(cagr(r), 1.0395 ** (252 / 4) - 1.0)

    def test_flat_series_zero(self):
        assert np.isclose(cagr(pd.Series([0.0] * 10)), 0.0)


class TestVolAndSharpe:
    def test_vol_is_annualized_std(self):
        r = pd.Series([0.01, -0.01, 0.02, 0.0], index=IDX4)
        assert np.isclose(annualized_vol(r), r.std() * np.sqrt(252))

    def test_sharpe_uses_excess(self):
        # Constant rf shifts the mean but not the std: sharpe(r, rf) must equal
        # sharpe(r - rf, 0) and differ from raw sharpe.
        r = pd.Series([0.01, -0.01, 0.02, 0.0], index=IDX4)
        rf = pd.Series(0.0001, index=IDX4)
        raw = r.mean() / r.std() * np.sqrt(252)
        excess = (r - 0.0001).mean() / (r - 0.0001).std() * np.sqrt(252)
        assert np.isclose(sharpe(r, rf), excess)
        assert not np.isclose(sharpe(r, rf), raw)


class TestSortino:
    def test_full_sample_downside_rms(self):
        # Negative parts: [0, -0.01, 0, 0] -> RMS over ALL 4 = sqrt(0.0001/4).
        r = pd.Series([0.01, -0.01, 0.02, 0.0], index=IDX4)
        downside = np.sqrt(0.01 ** 2 / 4)
        assert np.isclose(sortino(r, ZERO_RF),
                          r.mean() / downside * np.sqrt(252))

    def test_exceeds_sharpe_for_positively_skewed(self):
        r = pd.Series([0.05, -0.01, 0.05, -0.01, 0.05], index=pd.RangeIndex(5))
        rf = pd.Series(0.0, index=pd.RangeIndex(5))
        assert sortino(r, rf) > sharpe(r, rf)


class TestDrawdown:
    def test_known_answer(self):
        # Equity: 1.1, 0.99, 1.0395 -> trough 0.99 vs peak 1.1 = -0.10.
        r = pd.Series([0.10, -0.10, 0.05], index=pd.RangeIndex(3))
        assert np.isclose(max_drawdown(r), 0.99 / 1.1 - 1.0)

    def test_monotonic_up_is_zero(self):
        r = pd.Series([0.01, 0.02, 0.01], index=pd.RangeIndex(3))
        assert np.isclose(max_drawdown(r), 0.0)

    def test_calmar_nan_when_no_drawdown(self):
        r = pd.Series([0.01, 0.02, 0.01], index=pd.RangeIndex(3))
        assert np.isnan(calmar(r))

    def test_calmar_known_answer(self):
        r = pd.Series([0.10, -0.10, 0.05], index=pd.RangeIndex(3))
        assert np.isclose(calmar(r), cagr(r) / abs(max_drawdown(r)))


class TestTimeUnderwater:
    def test_longest_run(self):
        # Equity: 1.1, 0.99, 1.04, 1.14 -> underwater days 2 and 3 (1.04 < 1.1),
        # back above at day 4 -> longest run 2.
        r = pd.Series([0.10, -0.10, 0.05, 0.10], index=IDX4)
        assert time_underwater(r) == 2

    def test_never_underwater(self):
        assert time_underwater(pd.Series([0.01, 0.01])) == 0


class TestHitRate:
    def test_counts_strictly_positive(self):
        r = pd.Series([0.01, -0.01, 0.0, 0.02], index=IDX4)
        assert np.isclose(hit_rate(r), 0.5)   # zeros are not hits


class TestSummarize:
    def test_keys_and_consistency(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.0005, 0.01, 252))
        rf = pd.Series(0.0001, index=r.index)
        s = summarize(r, rf)
        assert np.isclose(s["cagr"], cagr(r))
        assert np.isclose(s["sharpe"], sharpe(r, rf))
        assert np.isclose(s["years"], 1.0)