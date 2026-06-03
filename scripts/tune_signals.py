"""Parameter tuning for momentum (lookback, skip_days) and inflation (component
weighting), via IC decay. EXPLORATORY — we interpret the grids for robustness,
not auto-pick the max IC (that would be data-snooping)."""
from __future__ import annotations

import yaml
import pandas as pd

from atlas.data.point_in_time import build_pit_macro
from atlas.evaluation.ic import compute_ic_decay
from atlas.signals.momentum import TSMomentum
from atlas.signals.macro_trend import InflationTrend

pd.set_option("display.float_format", lambda v: f"{v:.4f}")
pd.set_option("display.width", 140)


def main() -> None:
    with open("config/universe.yaml") as f:
        cfg = yaml.safe_load(f)
    prices = pd.read_parquet("data/etf_prices.parquet")
    fred_raw = pd.read_parquet("data/fred_raw.parquet")
    fred_vintages = pd.read_parquet("data/fred_vintages.parquet")
    macro = build_pit_macro(fred_raw, fred_vintages, prices, cfg)

    # ---- Momentum: lookback x skip_days grid (mean IC across horizons) ----
    print("=" * 78)
    print("  MOMENTUM TUNING — mean IC by (lookback, skip_days), across horizons")
    print("=" * 78)
    lookbacks = [21, 63, 126, 252]
    skips = [0, 5, 21]
    for skip in skips:
        print(f"\n--- skip_days = {skip} ---")
        rows = {}
        for lb in lookbacks:
            sig = TSMomentum(lookback_days=lb, skip_days=skip).compute(prices, macro)
            decay = compute_ic_decay(sig, prices, include_icir=False, min_assets=5)
            rows[lb] = decay["mean_ic"]
        grid = pd.DataFrame(rows)
        grid.columns.name = "lookback"
        grid.index.name = "horizon"
        print(grid)

    # ---- Inflation: which component carries the predictive power? ----
    print("\n" + "=" * 78)
    print("  INFLATION TUNING — IC by indicator component (full sample + 2021-23)")
    print("=" * 78)
    variants = {
        "equal_weight (3 realized + breakeven)": InflationTrend(),
        "realized_only (CPI/coreCPI/PCE)": InflationTrend(rate_indicators=()),
        "breakeven_only (T5YIE)": InflationTrend(index_indicators=()),
    }
    hi = slice("2021-01-01", "2023-06-30")
    for name, sig_obj in variants.items():
        sig = sig_obj.compute(prices, macro)
        decay = compute_ic_decay(sig, prices, include_icir=False, min_assets=5)
        # also the high-inflation-regime IC at 21d
        from atlas.data.returns import forward_returns
        from atlas.evaluation.ic import compute_ic
        fwd21 = forward_returns(prices, horizon=21)
        hi_ic = compute_ic(sig.loc[hi], fwd21.loc[hi], min_assets=5)
        print(f"\n--- {name} ---")
        print("full-sample mean IC by horizon:")
        print(decay["mean_ic"].to_frame().T.to_string(index=False))
        print(f"high-inflation 2021-2023 IC @21d: {hi_ic:.4f}")


if __name__ == "__main__":
    main()