"""Apply risk limits to sized weights and verify the caps on real data.

Run: uv run python scripts/build_risk.py
"""
from __future__ import annotations

import yaml
import pandas as pd

from atlas.data.point_in_time import build_pit_macro
from atlas.portfolio.roster import build_roster, ROSTER_SCALES
from atlas.portfolio.blend import blend_signals
from atlas.portfolio.sizing import compute_target_weights
from atlas.portfolio.risk import apply_risk_limits

pd.set_option("display.float_format", lambda v: f"{v:.4f}")


def max_bucket_frac(w, risk_buckets):
    g = w.abs().sum(axis=1)
    fr = pd.DataFrame(index=w.index)
    for b, tickers in risk_buckets.items():
        cols = [t for t in tickers if t in w.columns]
        fr[b] = w[cols].abs().sum(axis=1) / g
    return fr.max(axis=1)


def report(pre, post, risk_buckets, limits):
    held = pre.notna().any(axis=1)
    g_pre, g_post = pre.abs().sum(axis=1)[held], post.abs().sum(axis=1)[held]
    pos_pre = pre.abs().div(pre.abs().sum(axis=1), axis=0).max(axis=1)[held]
    pos_post = post.abs().div(post.abs().sum(axis=1), axis=0).max(axis=1)[held]
    buc_pre = max_bucket_frac(pre, risk_buckets)[held]
    buc_post = max_bucket_frac(post, risk_buckets)[held]
    pos_abs_post = post.abs().max(axis=1)[held]                    # absolute, not /gross
    buc_abs_post = pd.DataFrame(
        {b: post[[t for t in ts if t in post.columns]].abs().sum(axis=1)
         for b, ts in risk_buckets.items()}
    ).max(axis=1)[held]

    n = int(held.sum())
    binds = int((g_pre > limits["max_gross"] + 1e-9).sum())
    print(f"Days held: {n}")
    print(f"Gross:  pre  mean {g_pre.mean():.2f}  max {g_pre.max():.2f}")
    print(f"        post mean {g_post.mean():.2f}  max {g_post.max():.2f}  "
          f"(cap {limits['max_gross']})")
    print(f"Leverage cap binds: {binds}/{n} days ({100*binds/n:.1f}%)")
    print(f"Max position (|w|/gross): pre {pos_pre.max():.3f}  post {pos_post.max():.3f}  "
          f"(cap {limits['max_position']})")
    print(f"Max bucket (sum|w|/gross): pre {buc_pre.max():.3f}  post {buc_post.max():.3f}  "
          f"(cap {limits['max_bucket']})")
    print(f"Max position (ABSOLUTE): post {pos_abs_post.max():.3f}  "
          f"(cap {limits['max_position'] * limits['max_gross']:.2f})")
    print(f"Max bucket (ABSOLUTE):   post {buc_abs_post.max():.3f}  "
          f"(cap {limits['max_bucket'] * limits['max_gross']:.2f})")


def main() -> None:
    with open("config/universe.yaml") as f:
        cfg = yaml.safe_load(f)
    risk_buckets, limits = cfg["risk_buckets"], cfg["risk_limits"]

    prices = pd.read_parquet("data/etf_prices.parquet")
    prices_unadj = pd.read_parquet("data/etf_prices_unadjusted.parquet")
    dividends = pd.read_parquet("data/etf_dividends.parquet")
    fred_raw = pd.read_parquet("data/fred_raw.parquet")
    fred_vintages = pd.read_parquet("data/fred_vintages.parquet")
    macro = build_pit_macro(fred_raw, fred_vintages, prices, cfg)

    signals = build_roster(prices=prices, prices_unadjusted=prices_unadj,
                           dividends=dividends, macro=macro)
    composite = blend_signals(signals, scales=ROSTER_SCALES)

    for mode in ("long_short", "long_only"):
        pre = compute_target_weights(composite, prices, mode=mode,
                                     target_vol=0.10, vol_window=126)
        post = apply_risk_limits(pre, risk_buckets,
                                 max_position=limits["max_position"],
                                 max_bucket=limits["max_bucket"],
                                 max_gross=limits["max_gross"])
        print("=" * 70)
        print(f"  MODE: {mode}")
        print("=" * 70)
        report(pre, post, risk_buckets, limits)
        print()


if __name__ == "__main__":
    main()