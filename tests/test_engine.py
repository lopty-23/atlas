"""Tests for the backtest engine (atlas.backtest.engine).

Known-answer tests, hand-computed: the lag (targets at t earn from t+1), drift
arithmetic, turnover-vs-drifted-book, cost charging, the cash/financing leg,
NaN targets, and PIT (future price perturbation leaves past returns unchanged).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.backtest.engine import rebalance_dates, run_backtest


def _prices_from_returns(rets: pd.DataFrame) -> pd.DataFrame:
    return 100.0 * (1.0 + rets).cumprod()


def _const_targets(index, cols, row) -> pd.DataFrame:
    return pd.DataFrame([row] * len(index), index=index, columns=cols)


# Jan month-end, two Feb days, Feb month-end, one Mar day.
DATES = pd.to_datetime(
    ["2020-01-31", "2020-02-03", "2020-02-04", "2020-02-28", "2020-03-02"]
)
ZERO_RF = pd.Series(0.0, index=DATES)


class TestRebalanceDates:
    def test_last_trading_day_per_month(self):
        out = rebalance_dates(DATES, "ME")
        expected = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-02"])
        assert list(out) == list(expected)


class TestLag:
    def test_targets_earn_from_next_day(self):
        # Rebalance at day0 close; +10% on day1 must land on day1, day0 ~ 0.
        rets = pd.DataFrame({"A": [0.0, 0.10, 0.0, 0.0, 0.0]}, index=DATES)
        bt = run_backtest(_const_targets(DATES, ["A"], [1.0]),
                          _prices_from_returns(rets), ZERO_RF, cost_per_side=0.0)
        assert np.isclose(bt["returns"].iloc[0], 0.0)    # old (empty) book
        assert np.isclose(bt["returns"].iloc[1], 0.10)   # new book earns t+1
        assert np.isclose(bt["weights_held"].iloc[0, 0], 0.0)
        assert np.isclose(bt["weights_held"].iloc[1, 0], 1.0)


class TestDrift:
    def test_weights_drift_with_relative_returns(self):
        # 0.5/0.5 set day0; day1 A +10%, B 0% -> day2 held = [.5*1.1, .5]/1.05.
        rets = pd.DataFrame({"A": [0.0, 0.10, 0.0, 0.0, 0.0],
                             "B": [0.0, 0.00, 0.0, 0.0, 0.0]}, index=DATES)
        bt = run_backtest(_const_targets(DATES, ["A", "B"], [0.5, 0.5]),
                          _prices_from_returns(rets), ZERO_RF, cost_per_side=0.0)
        assert np.allclose(bt["weights_held"].iloc[2],
                           [0.5 * 1.1 / 1.05, 0.5 / 1.05])


class TestTurnoverAndCosts:
    def test_turnover_measured_against_drifted_book(self):
        # Drift to [.5238, .4762] by the Feb-28 rebalance; trading back to
        # [.5, .5] trades 2 * 0.02381. Against stale targets it would be 0.
        rets = pd.DataFrame({"A": [0.0, 0.10, 0.0, 0.0, 0.0],
                             "B": [0.0, 0.00, 0.0, 0.0, 0.0]}, index=DATES)
        bt = run_backtest(_const_targets(DATES, ["A", "B"], [0.5, 0.5]),
                          _prices_from_returns(rets), ZERO_RF, cost_per_side=0.0)
        drifted = np.array([0.5 * 1.1 / 1.05, 0.5 / 1.05])
        expected = np.abs(np.array([0.5, 0.5]) - drifted).sum()
        assert np.isclose(bt["turnover"].loc["2020-02-28"], expected)

    def test_cost_charged_on_rebalance_day_return(self):
        # Day0: trade 0 -> 1.0 of notional at 10bp -> day0 return = -0.001.
        rets = pd.DataFrame({"A": [0.0, 0.0, 0.0, 0.0, 0.0]}, index=DATES)
        bt = run_backtest(_const_targets(DATES, ["A"], [1.0]),
                          _prices_from_returns(rets), ZERO_RF, cost_per_side=0.001)
        assert np.isclose(bt["returns"].iloc[0], -0.001)
        assert np.isclose(bt["costs"].iloc[0], 0.001)


class TestCashLeg:
    def test_empty_book_earns_rf(self):
        rf = pd.Series(0.0001, index=DATES)
        rets = pd.DataFrame({"A": [0.0, 0.02, 0.0, 0.0, 0.0]}, index=DATES)
        targets = _const_targets(DATES, ["A"], [np.nan])     # NaN -> no position
        bt = run_backtest(targets, _prices_from_returns(rets), rf)
        assert np.allclose(bt["returns"], 0.0001)            # rf every day, no cost

    def test_half_invested_splits_asset_and_rf(self):
        # net 0.5: day1 = 0.5*0.02 + 0.5*rf.
        rf = pd.Series(0.0001, index=DATES)
        rets = pd.DataFrame({"A": [0.0, 0.02, 0.0, 0.0, 0.0],
                             "B": [0.0, 0.00, 0.0, 0.0, 0.0]}, index=DATES)
        bt = run_backtest(_const_targets(DATES, ["A", "B"], [0.5, 0.0]),
                          _prices_from_returns(rets), rf, cost_per_side=0.0)
        assert np.isclose(bt["returns"].iloc[1], 0.5 * 0.02 + 0.5 * 0.0001)

    def test_levered_book_pays_rf(self):
        # net 1.5: day1 = 1.5*0.02 - 0.5*rf (borrowing at rf).
        rf = pd.Series(0.0001, index=DATES)
        rets = pd.DataFrame({"A": [0.0, 0.02, 0.0, 0.0, 0.0]}, index=DATES)
        bt = run_backtest(_const_targets(DATES, ["A"], [1.5]),
                          _prices_from_returns(rets), rf, cost_per_side=0.0)
        assert np.isclose(bt["returns"].iloc[1], 1.5 * 0.02 - 0.5 * 0.0001)


class TestEquity:
    def test_equity_is_cumprod_of_returns(self):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2020-01-01", periods=120)
        rets = pd.DataFrame(rng.normal(0, 0.01, (120, 2)), index=idx,
                            columns=["A", "B"])
        bt = run_backtest(_const_targets(idx, ["A", "B"], [0.4, 0.3]),
                          _prices_from_returns(rets), pd.Series(0.0, index=idx))
        assert np.allclose(bt["equity"], (1.0 + bt["returns"]).cumprod())


class TestPointInTime:
    def test_future_price_perturbation_leaves_past_unchanged(self):
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2020-01-01", periods=150)
        rets = pd.DataFrame(rng.normal(0, 0.01, (150, 3)), index=idx,
                            columns=["A", "B", "C"])
        prices = _prices_from_returns(rets)
        targets = _const_targets(idx, ["A", "B", "C"], [0.3, 0.3, 0.2])
        rf = pd.Series(0.0001, index=idx)

        bt1 = run_backtest(targets, prices, rf)
        prices2 = prices.copy()
        prices2.iloc[-10:] *= 1.2
        bt2 = run_backtest(targets, prices2, rf)

        pd.testing.assert_series_equal(bt1["returns"].iloc[:-10],
                                       bt2["returns"].iloc[:-10])
        pd.testing.assert_frame_equal(bt1["weights_held"].iloc[:-10],
                                      bt2["weights_held"].iloc[:-10])