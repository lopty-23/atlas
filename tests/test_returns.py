from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.data.returns import (
    excess_returns,
    forward_returns,
    realized_volatility,
    simple_returns,
    trailing_returns,
)


def _step_prices() -> pd.DataFrame:
    """Prices that are flat, then jump +10% between day 2 and day 3."""
    return pd.DataFrame(
        {"X": [100.0, 100.0, 100.0, 110.0, 110.0, 110.0]},
        index=pd.bdate_range("2024-01-01", periods=6),
    )


class TestDirectionality:
    def test_forward_return_sees_future_jump(self) -> None:
        prices = _step_prices()
        fwd = forward_returns(prices, horizon=1)
        # Day index 2 looks ahead to day 3, where the +10% jump occurs.
        assert np.isclose(fwd["X"].iloc[2], 0.10), (
            "Forward return on day 2 should see the +10% jump that occurs on day 3"
        )
        
    def test_trailing_return_does_not_see_future_jump(self) -> None:
        prices = _step_prices()
        trl = trailing_returns(prices, lookback=1)
        # Day index 2 looks back to day 1, both flat → no jump yet.
        assert np.isclose(trl["X"].iloc[2], 0.0), (
            "Trailing return on day 2 must NOT see the future jump — that's look-ahead"
        )

    def test_trailing_return_sees_past_jump(self) -> None:
        prices = _step_prices()
        trl = trailing_returns(prices, lookback=1)
        # Day index 3 looks back to day 2 → captures the jump that just happened.
        assert np.isclose(trl["X"].iloc[3], 0.10), (
            "Trailing return on day 3 should capture the jump that just occurred"
        )

    def test_forward_equals_next_trailing(self) -> None:
        """forward_return(t) should equal trailing_return(t+1) — same event,
        different reference date."""
        prices = pd.DataFrame(
            {"X": [100.0, 101.0, 99.0, 102.0, 105.0]},
            index=pd.bdate_range("2024-01-01", periods=5),
        )
        fwd = forward_returns(prices, horizon=1)["X"]
        trl = trailing_returns(prices, lookback=1)["X"]
        # forward at position i == trailing at position i+1
        for i in range(len(prices) - 1):
            assert np.isclose(fwd.iloc[i], trl.iloc[i + 1]), (
                f"forward[{i}] should equal trailing[{i + 1}]"
            )


class TestNaNPlacement:
    def test_forward_nan_at_end(self) -> None:
        prices = _step_prices()
        fwd = forward_returns(prices, horizon=1)
        assert pd.isna(fwd["X"].iloc[-1]), "Forward return must be NaN on the last row"
        assert not pd.isna(fwd["X"].iloc[0]), "Forward return should exist on the first row"

    def test_trailing_nan_at_start(self) -> None:
        prices = _step_prices()
        trl = trailing_returns(prices, lookback=1)
        assert pd.isna(trl["X"].iloc[0]), "Trailing return must be NaN on the first row"
        assert not pd.isna(trl["X"].iloc[-1]), "Trailing return should exist on the last row"


class TestMagnitudes:
    def test_simple_return_values(self) -> None:
        prices = pd.DataFrame(
            {"X": [100.0, 110.0, 99.0]},
            index=pd.bdate_range("2024-01-01", periods=3),
        )
        r = simple_returns(prices)["X"]
        assert np.isclose(r.iloc[1], 0.10)        # 100 -> 110 = +10%
        assert np.isclose(r.iloc[2], -0.10)       # 110 -> 99  = -10%

    def test_multi_period_return(self) -> None:
        prices = pd.DataFrame(
            {"X": [100.0, 105.0, 110.0]},
            index=pd.bdate_range("2024-01-01", periods=3),
        )
        # 2-period return from 100 to 110 = +10%
        r = simple_returns(prices, periods=2)["X"]
        assert np.isclose(r.iloc[2], 0.10)


class TestExcessReturns:
    def test_excess_subtracts_risk_free(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=3)
        asset_rets = pd.DataFrame({"X": [0.001, 0.001, 0.001]}, index=dates)
        # 2.52% annual rate → 0.0001 daily (2.52/100/252)
        rf_annual = pd.Series([2.52, 2.52, 2.52], index=dates)
        excess = excess_returns(asset_rets, rf_annual)
        # 0.001 - 0.0001 = 0.0009
        assert np.allclose(excess["X"].values, 0.0009), (
            "Excess return should be asset return minus daily risk-free rate"
        )


class TestVolatility:
    def test_zero_vol_for_constant_returns(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=100)
        constant_rets = pd.DataFrame({"X": [0.001] * 100}, index=dates)
        vol = realized_volatility(constant_rets, window=20)
        # Constant returns have zero standard deviation
        assert np.isclose(vol["X"].iloc[-1], 0.0)

    def test_annualization_factor(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=300)
        rng = np.random.default_rng(42)
        rets = pd.DataFrame({"X": rng.normal(0, 0.01, 300)}, index=dates)
        vol_annual = realized_volatility(rets, window=252, annualize=True)["X"].iloc[-1]
        vol_daily = realized_volatility(rets, window=252, annualize=False)["X"].iloc[-1]
        # Annual vol should be daily vol times sqrt(252)
        assert np.isclose(vol_annual, vol_daily * np.sqrt(252))