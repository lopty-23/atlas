"""Signal correlation analysis: positioning similarity + IC co-movement.

Run: uv run python scripts/evaluate_correlations.py
"""
from __future__ import annotations

import yaml
import pandas as pd

from atlas.data.point_in_time import build_pit_macro
from atlas.evaluation.correlation import signal_value_correlation, ic_correlation
from atlas.signals.momentum import TSMomentum
from atlas.signals.carry import BondCarry, FXCarry
from atlas.signals.macro_trend import GrowthTrend, InflationTrend

pd.set_option("display.float_format", lambda v: f"{v:.3f}")
pd.set_option("display.width", 120)


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
        "TSMomentum": TSMomentum().compute(prices, macro),
        "BondCarry": BondCarry(dividends=dividends, prices_unadjusted=prices_unadj).compute(prices, macro),
        "FXCarry": FXCarry().compute(prices, macro),
        "GrowthTrend": GrowthTrend().compute(prices, macro),
        "InflationTrend": InflationTrend().compute(prices, macro),
    }
    min_assets_map = {"FXCarry": 4}

    print("=" * 70)
    print("  SIGNAL-VALUE CORRELATION (positioning similarity)")
    print("  NaN = signals cover disjoint assets (auto-diversifying)")
    print("=" * 70)
    print(signal_value_correlation(signals))
    print()

    for h in (21, 63, 126):
        print("=" * 70)
        print(f"  IC CORRELATION at {h}-day horizon (do edges co-occur?)")
        print("=" * 70)
        print(ic_correlation(signals, prices, horizon=h, min_assets_map=min_assets_map))
        print()


if __name__ == "__main__":
    main()