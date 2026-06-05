"""Position sizing: composite conviction -> vol-targeted portfolio weights.

Three stages, per date:
  A. Directional weights (mode fork). long_short: composite straight through
     (sign = direction; NO demean, so the net inflation tilt is retained).
     long_only: shorts clipped to zero (the long-short book minus its short leg).
  B. Inverse-vol scaling -- w_i ∝ dir_i / sigma_i, so each asset contributes
     comparable risk (bond vol ~5%, equity ~16%, commodity ~25%); without it,
     high-vol assets dominate.
  C. Portfolio vol-targeting -- scale the whole book by a per-date scalar so its
     ex-ante annualized vol hits a target (~10%). The scalar uses ONLY trailing
     data (current weights on PAST returns) -- the look-ahead-sensitive step.

Output: a DAILY date x asset target-weight frame, gross-UNCONSTRAINED (sum|w|
may exceed 1 -- vol-targeting demands leverage when trailing vol < target).
Leverage caps live in risk.py; rebalance timing and the execution lag live in
the backtest (weights_today = f(signals_yesterday)). Sizing's output at t uses
data through t; the backtest shifts it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.data.returns import simple_returns, realized_volatility

TRADING_DAYS = 252


def _directional_weights(composite: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "long_short":
        return composite.copy()
    if mode == "long_only":
        return composite.clip(lower=0.0)  # clip leaves NaN as NaN (NaN is not < 0)
    raise ValueError(f"mode must be 'long_short' or 'long_only', got {mode!r}")


def _inverse_vol_scale(
    directional: pd.DataFrame, asset_vol: pd.DataFrame
) -> pd.DataFrame:
    return directional / asset_vol


def _vol_target_scalar(
    ivol_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    target_vol: float,
    window: int,
) -> pd.Series:
    """Stage C: per-date scalar k_t so the scaled book's ex-ante vol == target."""
    cols = list(ivol_weights.columns)
    w = ivol_weights.fillna(0.0)  # absent/warm-up asset -> 0 contribution

    # Rolling covariance panel: rows MultiIndex (date, asset_i), columns asset_j.
    # min_periods defaults to `window`, so cov is NaN until a full window exists.
    roll_cov = asset_returns.rolling(window=window).cov()

    # Broadcast w across the asset_i row level: at row (t, i), column j must hold
    # w[t, j]. The weight on column j depends only on the date t, not on i, so we
    # repeat each date's weight row once per asset_i.
    dates = roll_cov.index.get_level_values(0)
    w_bcast = w.reindex(index=dates)      # each date's row repeated (in row order)
    w_bcast.index = roll_cov.index        # positional realign onto the MultiIndex

    # Step 1 -- inner[(t, i)] = sum_j Cov_t(i, j) * w[t, j] = (Sigma_t w_t)[i].
    # skipna drops NaN cov entries; those correspond to zero-weight (warm-up /
    # insufficient-history) assets whose true contribution is 0, so dropping them
    # is exact, not an approximation.
    inner = (roll_cov * w_bcast).sum(axis=1)

    # Step 2 -- port_var_t = sum_i w[t, i] * inner[(t, i)] = w_t^T Sigma_t w_t.
    inner_mat = inner.unstack().reindex(columns=cols)   # date x asset_i
    port_var = (inner_mat * w).sum(axis=1)

    port_vol = np.sqrt(port_var) * np.sqrt(TRADING_DAYS)

    # Poison a zero/NaN denominator to NaN BEFORE dividing (same guard as blend):
    # an empty or unestimable book -> NaN scalar -> NaN weights ("no position"),
    # which the backtest credits as cash. Avoids the 0/0 and inf cases entirely.
    return target_vol / port_vol.where(port_vol > 0)


def compute_target_weights(
    composite: pd.DataFrame,
    prices: pd.DataFrame,
    mode: str = "long_short",
    target_vol: float = 0.10,
    vol_window: int = 126,
) -> pd.DataFrame:
    prices = prices.reindex(index=composite.index, columns=composite.columns)
    asset_returns = simple_returns(prices)
    asset_vol = realized_volatility(asset_returns, window=vol_window, annualize=True)

    directional = _directional_weights(composite, mode)
    ivol_weights = _inverse_vol_scale(directional, asset_vol)
    k = _vol_target_scalar(ivol_weights, asset_returns, target_vol, vol_window)

    # Final weights use the ORIGINAL ivol_weights (with NaN), so warm-up/abstain
    # cells stay NaN regardless of k.
    return ivol_weights.mul(k, axis=0)