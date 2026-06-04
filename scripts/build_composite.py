"""Build the 4-signal composite and verify the blend on real data.

Confirms blend_signals integrates with the real
signal stack before sizing consumes the composite. Checks the two subtle
behaviors on real data -- per-date renormalization over PRESENT signals (union
grid; signals with different start dates / asset coverage) and the
InflationTrend rescale.
"""
from __future__ import annotations

import yaml
import pandas as pd

from atlas.data.point_in_time import build_pit_macro
from atlas.portfolio.roster import build_roster, ROSTER_SCALES
from atlas.portfolio.blend import blend_signals

pd.set_option("display.float_format", lambda v: f"{v:.4f}")

SPOT_CHECKS = [("SPY", "2004-06-01"), ("IEF", "2022-06-01")]


def main() -> None:
    with open("config/universe.yaml") as f:
        cfg = yaml.safe_load(f)

    prices = pd.read_parquet("data/etf_prices.parquet")
    prices_unadj = pd.read_parquet("data/etf_prices_unadjusted.parquet")
    dividends = pd.read_parquet("data/etf_dividends.parquet")
    fred_raw = pd.read_parquet("data/fred_raw.parquet")
    fred_vintages = pd.read_parquet("data/fred_vintages.parquet")
    macro = build_pit_macro(fred_raw, fred_vintages, prices, cfg)

    # 4-signal roster (GrowthTrend dropped). BondCarry takes dividends +
    # unadjusted prices via constructor, exactly as in evaluate_signals.py.
    signals = build_roster(
        prices=prices,
        prices_unadjusted=prices_unadj,
        dividends=dividends,
        macro=macro,
    )
    
    composite = blend_signals(signals, scales=ROSTER_SCALES)
    print("dtypes:", {n: str(s.values.dtype) for n, s in signals.items()},
          "| composite:", composite.values.dtype)
    aligned = {name: signals[name].reindex_like(composite) for name in signals}

    # --- Shape & coverage ------------------------------------------------
    print("=" * 70)
    print("  COMPOSITE SHAPE & COVERAGE")
    print("=" * 70)
    print(f"Date range : {composite.index.min().date()} -> {composite.index.max().date()}")
    print(f"Assets     : {composite.shape[1]}")
    print(f"Avg scored per date: {composite.notna().sum(axis=1).mean():.1f}")

    # --- Renormalization proof: how many signals fire per scored cell ----
    # Max is 3, never 4: no asset is in BOTH the bond-carry and FX-carry
    # universes. A spread across 1/2/3 means renorm is doing real work.
    present = sum(aligned[name].notna().astype(int) for name in signals)
    counts = present.values[composite.notna().values]
    print("\nSignals present per scored cell:")
    print(pd.Series(counts).value_counts().sort_index().to_string())

    # --- Hand-checkable spot checks --------------------------------------
    # Verify the INVARIANT (composite == renormalized mean of present signals,
    # inflation taken at its /2 value), not a hardcoded number.
    print("\n" + "=" * 70)
    print("  SPOT CHECKS  (composite == mean of present signals)")
    print("=" * 70)
    for asset, date in SPOT_CHECKS:
        row = composite.index.asof(pd.Timestamp(date))
        print(f"\n{asset} @ {row.date()}:")
        for name in signals:
            scale = ROSTER_SCALES.get(name, 1.0)
            v = aligned[name].loc[row, asset]
            note = f"   (/{scale:g} = {v / scale:.4f})" if scale != 1.0 else ""
            print(f"  {name:15s}: {v:8.4f}{note}")
        print(f"  {'composite':15s}: {composite.loc[row, asset]:8.4f}")


if __name__ == "__main__":
    main()