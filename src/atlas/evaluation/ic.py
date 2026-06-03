"""Information Coefficient (IC): how well a signal predicts forward returns.

This module is signal-agnostic: it works on any date x asset score frame,
which is what every Signal.compute() returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from atlas.data.returns import forward_returns

# Minimum assets with both a signal and a return on a date, for that date's
# cross-sectional correlation to be meaningful. Below this, the date is skipped.
TRADING_DAYS = 252
MIN_ASSETS_PER_DATE = 5


def compute_ic_timeseries(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    method: str = "spearman",
    min_assets: int = MIN_ASSETS_PER_DATE,
    lag: int = 1,
) -> pd.Series:
    """Per-date cross-sectional IC between a (lagged) signal and forward returns.

    On each date, compute the rank correlation across assets between the
    signal scores and the forward returns. Returns one IC value per date
    (NaN on dates with fewer than `min_assets` valid pairs).
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
    if lag < 0:
        raise ValueError(f"lag must be non-negative, got {lag}")

    lagged_signal = signal.shift(lag)

    common_cols = lagged_signal.columns.intersection(fwd_returns.columns)
    sig = lagged_signal[common_cols]
    ret = fwd_returns.reindex(index=lagged_signal.index, columns=common_cols)

    ics: dict[pd.Timestamp, float] = {}
    for date in sig.index:
        s_row = sig.loc[date]
        r_row = ret.loc[date]
        valid = s_row.notna() & r_row.notna()
        if valid.sum() < min_assets:
            ics[date] = np.nan
            continue
        s_valid = s_row[valid]
        r_valid = r_row[valid]
        if s_valid.nunique() < 2 or r_valid.nunique() < 2:
            ics[date] = np.nan
            continue
        ic = s_valid.corr(r_valid, method=method)
        ics[date] = ic

    return pd.Series(ics, name="ic")


def compute_ic(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    method: str = "spearman",
    min_assets: int = MIN_ASSETS_PER_DATE,
    lag: int = 1,
) -> float:
    ic_ts = compute_ic_timeseries(
        signal, fwd_returns, method=method, min_assets=min_assets, lag=lag
    )
    return ic_ts.mean()

def compute_rolling_ic(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    window: int = 252,
    method: str = "spearman",
    min_assets: int = MIN_ASSETS_PER_DATE,
    lag: int = 1,
) -> pd.Series:
    ic_ts = compute_ic_timeseries(
        signal, fwd_returns, method=method, min_assets=min_assets, lag=lag
    )
    return ic_ts.rolling(window=window, min_periods=window // 2).mean()


def _newey_west_se(x: pd.Series, n_lags: int) -> float:
    """Newey-West (HAC) standard error of the MEAN of an autocorrelated series.

    With overlapping forward returns, consecutive IC observations are
    positively autocorrelated, so the naive standard error of their mean is
    understated. Newey-West inflates the variance estimate using the series'
    own autocovariances out to `n_lags`, with Bartlett (triangular) weights,
    giving an honest standard error.

    Returns the standard error of the sample mean (not the series std).
    """
    x = x.dropna()
    n = len(x)
    if n < 2:
        return float("nan")

    demeaned = (x - x.mean()).values
    # Lag 0 autocovariance (the usual variance).
    gamma0 = np.dot(demeaned, demeaned) / n
    var = gamma0

    # Add autocovariance terms with Bartlett weights w_k = 1 - k/(n_lags+1).
    n_lags = min(n_lags, n - 1)
    for k in range(1, n_lags + 1):
        weight = 1.0 - k / (n_lags + 1)
        gamma_k = np.dot(demeaned[k:], demeaned[:-k]) / n
        var += 2.0 * weight * gamma_k

    # Variance of the mean = long-run variance / n. SE = sqrt of that.
    var_of_mean = var / n
    if var_of_mean <= 0:
        return float("nan")
    return float(np.sqrt(var_of_mean))


def compute_icir(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    horizon: int,
    method: str = "spearman",
    min_assets: int = MIN_ASSETS_PER_DATE,
    lag: int = 1,
    annualize: bool = True,
) -> dict[str, float]:
    """IC information ratio: mean IC divided by its (HAC-corrected) volatility."""
    ic_ts = compute_ic_timeseries(
        signal, fwd_returns, method=method, min_assets=min_assets, lag=lag
    ).dropna()

    n = len(ic_ts)
    if n < 2:
        return {"mean_ic": float("nan"), "ic_se": float("nan"),
                "icir": float("nan"), "t_stat": float("nan"), "n_obs": float(n)}

    mean_ic = ic_ts.mean()
    naive_std = ic_ts.std()

    # Conventional ICIR (mean / per-period std), optionally annualized.
    icir = mean_ic / naive_std if naive_std > 0 else float("nan")
    if annualize and not np.isnan(icir):
        icir *= np.sqrt(TRADING_DAYS / horizon)

    # HAC standard error of the mean -> honest significance t-stat.
    nw_se = _newey_west_se(ic_ts, n_lags=horizon)
    t_stat = mean_ic / nw_se if nw_se and nw_se > 0 else float("nan")

    return {
        "mean_ic": float(mean_ic),
        "ic_se": float(nw_se),
        "icir": float(icir),
        "t_stat": float(t_stat),
        "n_obs": float(n),
    }

DEFAULT_DECAY_HORIZONS = (1, 5, 21, 63, 126, 252)


def compute_ic_decay(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_DECAY_HORIZONS,
    method: str = "spearman",
    min_assets: int = MIN_ASSETS_PER_DATE,
    lag: int = 1,
    include_icir: bool = True,
) -> pd.DataFrame:
    """IC (and ICIR) at a range of forward horizons — the decay curve.

    For each horizon h, build h-day forward returns from `prices` and compute
    the mean IC of the signal against them. The shape of mean IC vs horizon
    reveals the signal's natural holding period: trend signals peak at longer
    horizons (63-252d), reversal signals at short ones (1-5d), and slow macro
    signals may only show meaningful IC at longer horizons.
    """
    rows: dict[int, dict[str, float]] = {}

    for h in horizons:
        fwd = forward_returns(prices, horizon=h)

        if include_icir:
            stats = compute_icir(
                signal, fwd, horizon=h, method=method,
                min_assets=min_assets, lag=lag, annualize=True,
            )
            rows[h] = {
                "mean_ic": stats["mean_ic"],
                "icir": stats["icir"],
                "t_stat": stats["t_stat"],
                "n_obs": stats["n_obs"],
            }
        else:
            mean_ic = compute_ic(
                signal, fwd, method=method, min_assets=min_assets, lag=lag
            )
            rows[h] = {"mean_ic": mean_ic}

    decay = pd.DataFrame.from_dict(rows, orient="index")
    decay.index.name = "horizon"
    return decay