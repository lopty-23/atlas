"""Backtest engine: daily target weights -> realized portfolio returns.

Conventions (the promissory notes from the portfolio layer, honored here):
- Execution lag: targets read at rebalance date t are traded at t's close and
  earn returns from t+1 (weights_today = f(data through yesterday)).
- Drift: between rebalances, positions (not weights) are held -- weights drift
  with relative returns. Turnover at a rebalance is measured against the
  DRIFTED book, not last rebalance's targets.
- Costs: turnover * cost_per_side, charged on the rebalance day's return.
- Cash/financing: residual (1 - net) earns/pays rf each day. Borrowing at rf,
  no short-borrow fees -- research convention, flatters long-short financing
  (flagged in DECISIONS). No drawdown throttle in v1 (deferred overlay).
NaN target = no position (0). Same-close execution is an idealization; costs
carry the realism burden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "ME") -> pd.DatetimeIndex:
    """Last trading day of each period in `index` (default month-end)."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period(freq[0])).last().values)


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    rf: pd.Series,
    rebalance: str = "ME",
    cost_per_side: float = 0.0001,
) -> dict:
    
    prices = prices.reindex(index=weights.index, columns=weights.columns)
    rets = prices.pct_change()
    # Safe only because a NaN return can only carry weight 0 (sizing masks
    # warm-ups/abstentions; targets are fillna(0) at trade time).
    rets_f = rets.fillna(0.0).to_numpy()
    targets = weights.fillna(0.0).to_numpy()
    rf_v = rf.reindex(weights.index).ffill().fillna(0.0).to_numpy()

    dates = weights.index
    rebal = set(rebalance_dates(dates, rebalance))
    n_days, n_assets = targets.shape

    w_held = np.zeros(n_assets)                  # book held during the day
    port_ret = np.zeros(n_days)
    turnover = np.zeros(n_days)
    costs = np.zeros(n_days)
    held_out = np.zeros((n_days, n_assets))

    for s in range(n_days):
        held_out[s] = w_held
        r = rets_f[s]
        net = w_held.sum()
        day_ret = w_held @ r + (1.0 - net) * rf_v[s]

        # Drift: each position grows with its own return; renormalize by the
        # book's growth so weights stay fractions of current value.
        if not np.isclose(day_ret, -1.0):
            w_held = w_held * (1.0 + r) / (1.0 + day_ret)

        if dates[s] in rebal:
            new_w = targets[s]
            traded = np.abs(new_w - w_held).sum()
            cost = traded * cost_per_side
            turnover[s] = traded
            costs[s] = cost
            day_ret -= cost                      # paid at the trade's close
            w_held = new_w

        port_ret[s] = day_ret

    returns = pd.Series(port_ret, index=dates, name="portfolio_return")
    return {
        "returns": returns,
        "equity": (1.0 + returns).cumprod(),
        "weights_held": pd.DataFrame(held_out, index=dates, columns=weights.columns),
        "turnover": pd.Series(turnover, index=dates),
        "costs": pd.Series(costs, index=dates),
    }