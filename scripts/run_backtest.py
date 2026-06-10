"""Run the full stack (blend -> size -> risk -> backtest) for both modes.

Run: uv run python scripts/run_backtest.py
Sanity checks only -- performance metrics wait for performance.py.
"""
from __future__ import annotations

import yaml
import numpy as np
import pandas as pd

from atlas.data.point_in_time import build_pit_macro
from atlas.portfolio.roster import build_roster, ROSTER_SCALES
from atlas.portfolio.blend import blend_signals
from atlas.portfolio.sizing import compute_target_weights
from atlas.portfolio.risk import apply_risk_limits
from atlas.backtest.engine import run_backtest

pd.set_option("display.float_format", lambda v: f"{v:.4f}")
TRADING_DAYS = 252


def main() -> None:
    with open("config/universe.yaml") as f:
        cfg = yaml.safe_load(f)
    prices = pd.read_parquet("data/etf_prices.parquet")
    prices_unadj = pd.read_parquet("data/etf_prices_unadjusted.parquet")
    dividends = pd.read_parquet("data/etf_dividends.parquet")
    fred_raw = pd.read_parquet("data/fred_raw.parquet")
    fred_vintages = pd.read_parquet("data/fred_vintages.parquet")
    macro = build_pit_macro(fred_raw, fred_vintages, prices, cfg)

    signals = build_roster(prices=prices, prices_unadjusted=prices_unadj,
                           dividends=dividends, macro=macro)
    composite = blend_signals(signals, scales=ROSTER_SCALES)
    rf = macro["DGS3MO"] / 100.0 / TRADING_DAYS   # daily decimal, PIT-lagged

    for mode in ("long_short", "long_only"):
        sized = compute_target_weights(composite, prices, mode=mode,
                                       target_vol=0.10, vol_window=126)
        capped = apply_risk_limits(sized, cfg["risk_buckets"],
                                   **cfg["risk_limits"])
        bt = run_backtest(capped, prices, rf)

        r = bt["returns"]
        live = bt["weights_held"].abs().sum(axis=1) > 0
        r_live = r[live]
        years = len(r_live) / TRADING_DAYS
        ann_to = bt["turnover"].sum() / years
        cost_drag = bt["costs"].sum() / years

        print("=" * 70)
        print(f"  MODE: {mode}")
        print("=" * 70)
        print(f"Live days        : {int(live.sum())}  "
              f"({r_live.index.min().date()} -> {r_live.index.max().date()})")
        print(f"Realized ann vol : {r_live.std() * np.sqrt(TRADING_DAYS):.4f}  (target 0.10)")
        print(f"Equity multiple  : {bt['equity'].iloc[-1]:.3f}x over {years:.1f}y")
        print(f"Ann turnover     : {ann_to:.2f}x gross/year")
        print(f"Ann cost drag    : {cost_drag * 1e4:.1f} bps/year")
        print()


if __name__ == "__main__":
    main()