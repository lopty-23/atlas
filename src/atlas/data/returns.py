"""Return computations: simple, log, excess, trailing, and forward."""
from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def simple_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return prices.pct_change(periods=periods)


def log_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return np.log(prices / prices.shift(periods))


def forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    return prices.shift(-horizon) / prices - 1.0


def trailing_returns(prices: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    return prices.pct_change(periods=lookback)


def excess_returns(
    asset_returns: pd.DataFrame,
    risk_free_annual: pd.Series,
) -> pd.DataFrame:
    rf_daily = (risk_free_annual / 100.0) / TRADING_DAYS
    rf_aligned = rf_daily.reindex(asset_returns.index, method="ffill")
    return asset_returns.sub(rf_aligned, axis=0)


def realized_volatility(
    returns: pd.DataFrame,
    window: int = 63,
    annualize: bool = True,
) -> pd.DataFrame:
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol


def annualize_return(daily_return: float, periods: int = TRADING_DAYS) -> float:
    return (1.0 + daily_return) ** periods - 1.0


def distribution_yield(
    dividends: pd.DataFrame,
    prices: pd.DataFrame,
    window_days: int = 365,
) -> pd.DataFrame:
    """Trailing-window distribution yield, aligned to the price calendar.

    yield[t, etf] = sum(dividends in (t - window_days, t]) / price[t, etf]"""
    common = sorted(set(dividends.columns) & set(prices.columns))
    if not common:
        return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    # Reindex dividends onto the price calendar, treating absent dates as 0 so the rolling sum is well-defined on every trading day.
    daily_divs = dividends[common].reindex(prices.index).fillna(0.0)
    trailing = daily_divs.rolling(f"{window_days}D").sum()

    out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    out[common] = trailing / prices[common]
    return out