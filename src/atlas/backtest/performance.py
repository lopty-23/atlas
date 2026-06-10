"""Performance metrics: pure functions of a daily portfolio-return series.

All metrics score exactly the series given -- pass the LIVE slice (engine
returns include flat pre-live zeros, which would dilute every annualized
number). Sharpe/Sortino use EXCESS returns over rf (raw-return Sharpe is
misleading for a levered book). No benchmark logic here; comparisons are a
script's job.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(returns: pd.Series) -> float:
    equity = (1.0 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return equity ** (1.0 / years) - 1.0


def annualized_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe(returns: pd.Series, rf: pd.Series) -> float:
    excess = returns - rf.reindex(returns.index).ffill().fillna(0.0)
    return excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)


def sortino(returns: pd.Series, rf: pd.Series) -> float:
    """Downside deviation = RMS of the negative excess returns (full-sample
    denominator, the standard convention)."""
    excess = returns - rf.reindex(returns.index).ffill().fillna(0.0)
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())
    return excess.mean() / downside * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss of the equity curve. Negative number."""
    equity = (1.0 + returns).cumprod()
    return (equity / equity.cummax() - 1.0).min()


def calmar(returns: pd.Series) -> float:
    dd = max_drawdown(returns)
    return np.nan if np.isclose(dd, 0.0) else cagr(returns) / abs(dd)


def time_underwater(returns: pd.Series) -> int:
    """Longest stretch (trading days) below the running equity peak."""
    equity = (1.0 + returns).cumprod()
    under = equity < equity.cummax()
    longest = current = 0
    for u in under:
        current = current + 1 if u else 0
        longest = max(longest, current)
    return longest


def hit_rate(returns: pd.Series) -> float:
    return (returns > 0).mean()


def summarize(returns: pd.Series, rf: pd.Series) -> dict:
    """All metrics in one dict. `returns` must be the live slice."""
    return {
        "cagr": cagr(returns),
        "ann_vol": annualized_vol(returns),
        "sharpe": sharpe(returns, rf),
        "sortino": sortino(returns, rf),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns),
        "time_underwater_days": time_underwater(returns),
        "hit_rate": hit_rate(returns),
        "years": len(returns) / TRADING_DAYS,
    }