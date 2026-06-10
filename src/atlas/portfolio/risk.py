"""Risk limits: cap sizing's gross-unconstrained weights into a tradeable book.

Three static caps applied per date, in order: per-asset, per-bucket, leverage.
Per-asset / per-bucket caps are fractions of TARGET gross (max_gross) -> fixed
ABSOLUTE thresholds, independent of the book's realized gross or concentration
(a fraction-of-realized-gross cap is infeasible when one name dominates the
book). Caps only REDUCE (|w_out| <= |w_in|); freed capital goes to cash.
Leverage is last: uniform whole-book scaling only reduces |w|, so it cannot push
any name/bucket back over its fixed cap. Applied once each, not iterated. See
DECISIONS.md for rationale and the gross-denominator choice.

Out of scope (live in the backtest): execution lag, rebalance timing, and the
path-dependent drawdown throttle.
"""
from __future__ import annotations

import pandas as pd


def _verify_bucket_coverage(
    assets: pd.Index, risk_buckets: dict[str, list[str]]
) -> None:
    """Assert every traded asset is in exactly one bucket (else it escapes the
    bucket cap silently). Extra config tickers not traded are ignored."""
    counts: dict[str, int] = {}
    for tickers in risk_buckets.values():
        for t in tickers:
            counts[t] = counts.get(t, 0) + 1

    missing = [a for a in assets if a not in counts]
    if missing:
        raise ValueError(
            f"assets not assigned to any risk bucket: {missing}. "
            "Add them to risk_buckets in universe.yaml."
        )
    duplicated = [t for t, n in counts.items() if n > 1]
    if duplicated:
        raise ValueError(f"assets in more than one risk bucket: {duplicated}.")


def _cap_positions(weights: pd.DataFrame, position_limit: float) -> pd.DataFrame:
    """Cap 1: trim each name to +-position_limit (absolute). Single-name op, so
    clip == scale."""
    return weights.clip(lower=-position_limit, upper=position_limit)


def _cap_buckets(
    weights: pd.DataFrame, risk_buckets: dict[str, list[str]], bucket_limit: float
) -> pd.DataFrame:
    """Cap 2: scale each over-cap bucket down to bucket_limit (absolute),
    preserving within-bucket ratios. Under-cap buckets untouched. NaN preserved."""
    out = weights.copy()
    for tickers in risk_buckets.values():
        cols = [t for t in tickers if t in out.columns]
        if not cols:
            continue
        bucket_abs = out[cols].abs().sum(axis=1)
        over = bucket_abs > bucket_limit
        factor = pd.Series(1.0, index=weights.index)
        factor[over] = bucket_limit / bucket_abs[over]
        out[cols] = out[cols].mul(factor, axis=0)
    return out


def _cap_leverage(weights: pd.DataFrame, max_gross: float) -> pd.DataFrame:
    """Cap 3: if gross > max_gross, scale the whole book down. Uniform scaling
    only reduces |w|, so it cannot re-violate caps 1-2."""
    gross = weights.abs().sum(axis=1)
    over = gross > max_gross
    factor = pd.Series(1.0, index=weights.index)
    factor[over] = max_gross / gross[over]
    return weights.mul(factor, axis=0)


def apply_risk_limits(
    weights: pd.DataFrame,
    risk_buckets: dict[str, list[str]],
    max_position: float = 0.20,
    max_bucket: float = 0.50,
    max_gross: float = 3.0,
) -> pd.DataFrame:
   
    _verify_bucket_coverage(weights.columns, risk_buckets)
    position_limit = max_position * max_gross
    bucket_limit = max_bucket * max_gross
    w = _cap_positions(weights, position_limit)
    w = _cap_buckets(w, risk_buckets, bucket_limit)
    w = _cap_leverage(w, max_gross)
    return w