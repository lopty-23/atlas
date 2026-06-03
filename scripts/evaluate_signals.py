"""Evaluate all five signals with the IC toolkit: mean IC, decay, ICIR, quintiles.

Run: uv run python scripts/evaluate_signals.py
This is exploratory analysis, not library code — it loads cached data, builds
each signal, and prints its predictive diagnostics for interpretation.
"""
from __future__ import annotations

import yaml
import pandas as pd

from atlas.data.point_in_time import build_pit_macro
from atlas.evaluation.ic import (
    compute_ic_decay,
    compute_icir,
    compute_quintile_returns,
)
from atlas.data.returns import forward_returns
from atlas.signals.momentum import TSMomentum
from atlas.signals.carry import BondCarry, FXCarry
from atlas.signals.macro_trend import GrowthTrend, InflationTrend

pd.set_option("display.float_format", lambda v: f"{v:.4f}")


def main() -> None:
    with open("config/universe.yaml") as f:
        cfg = yaml.safe_load(f)

    prices = pd.read_parquet("data/etf_prices.parquet")
    prices_unadj = pd.read_parquet("data/etf_prices_unadjusted.parquet")
    dividends = pd.read_parquet("data/etf_dividends.parquet")
    fred_raw = pd.read_parquet("data/fred_raw.parquet")
    fred_vintages = pd.read_parquet("data/fred_vintages.parquet")
    macro = build_pit_macro(fred_raw, fred_vintages, prices, cfg)

    signals = {
        "TSMomentum": (TSMomentum().compute(prices, macro), 5),
        "BondCarry": (
            BondCarry(dividends=dividends, prices_unadjusted=prices_unadj)
            .compute(prices, macro),
            5,
        ),
        "FXCarry": (FXCarry().compute(prices, macro), 4),
        "GrowthTrend": (GrowthTrend().compute(prices, macro), 5),
        "InflationTrend": (InflationTrend().compute(prices, macro), 5),
    }

    for name, (sig, min_assets) in signals.items():
        print("=" * 70)
        print(f"  {name}  (min_assets={min_assets})")
        print("=" * 70)

        avg_coverage = sig.notna().sum(axis=1).mean()
        print(f"Avg assets scored per date: {avg_coverage:.1f}")

        decay = compute_ic_decay(sig, prices, min_assets=min_assets)
        print("\nIC decay (mean IC / annualized ICIR / HAC t-stat):")
        print(decay[["mean_ic", "icir", "t_stat", "n_obs"]])

        fwd21 = forward_returns(prices, horizon=21)
        q = compute_quintile_returns(sig, fwd21, n_buckets=5, min_assets=min_assets)
        print(f"\nQuintile (21d fwd) spread (top - bottom): {q['spread']:.4f}")
        print("Bucket mean fwd returns:")
        print(q["bucket_returns"])
        print()


if __name__ == "__main__":
    main()