"""Build target weights for both modes and verify sizing on real data.

Run: uv run python scripts/build_sizing.py
"""
from __future__ import annotations

import yaml
import numpy as np
import pandas as pd

from atlas.data.point_in_time import build_pit_macro
from atlas.data.returns import simple_returns
from atlas.portfolio.roster import build_roster, ROSTER_SCALES
from atlas.portfolio.blend import blend_signals
from atlas.portfolio.sizing import compute_target_weights

pd.set_option("display.float_format", lambda v: f"{v:.4f}")
TRADING_DAYS = 252
WINDOW = 126


def independent_trailing_vol(weights, asset_returns, window):
    """Matrix-FREE trailing portfolio vol under fixed current weights.

    Independent cross-check of sizing's w^T Sigma w: at each date t, synthesize
    the portfolio return on each past day (weights fixed at t), take std over the
    trailing window. If sizing's contraction is correct this ~= target (0.10)
    where weights exist. Per-date loop is fine here (diagnostic, not library).
    """
    cols = list(weights.columns)
    rets = asset_returns.reindex(columns=cols).to_numpy()
    W = weights.reindex(columns=cols).to_numpy()
    idx = weights.index
    out = np.full(len(idx), np.nan)
    for t in range(window - 1, len(idx)):
        wt = W[t]
        if not np.isfinite(wt).any():
            continue
        mask = ~np.isfinite(wt)
        wt = np.where(mask, 0.0, wt)
        win = rets[t - window + 1 : t + 1].copy()
        win[:, mask] = 0.0                 # abstaining assets carry weight 0
        port = win @ wt
        if np.isnan(port).any():
            continue
        out[t] = port.std(ddof=1) * np.sqrt(TRADING_DAYS)
    return pd.Series(out, index=idx)


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
    aligned = prices.reindex(index=composite.index, columns=composite.columns)
    asset_returns = simple_returns(aligned)
    asset_vol = (asset_returns.rolling(WINDOW).std() * np.sqrt(TRADING_DAYS)).mean()

    for mode in ("long_short", "long_only"):
        W = compute_target_weights(composite, prices, mode=mode,
                                   target_vol=0.10, vol_window=WINDOW)
        held = W.notna().any(axis=1)
        gross = W.abs().sum(axis=1)
        net = W.sum(axis=1)

        print("=" * 70)
        print(f"  MODE: {mode}")
        print("=" * 70)
        print(f"Weights start : {held.idxmax().date()}  (warm-up before -> NaN)")
        print(f"Avg assets held: {W.notna().sum(axis=1)[held].mean():.1f}")
        print(f"\nGross leverage (sum|w|):  mean {gross[held].mean():.2f}  "
              f"median {gross[held].median():.2f}  "
              f"p95 {gross[held].quantile(0.95):.2f}  max {gross[held].max():.2f}")
        print(f"Net exposure (sum w):     mean {net[held].mean():+.2f}  "
              f"min {net[held].min():+.2f}  max {net[held].max():+.2f}")

        ivol = independent_trailing_vol(W, asset_returns, WINDOW)
        print(f"\nFinal book trailing vol (independent check, target 0.10): "
              f"mean {ivol[held].mean():.4f}  median {ivol[held].median():.4f}")
        print()

    # Inverse-vol behavior: avg |weight| per asset vs that asset's vol. Expect
    # LOW-vol assets (bonds) to carry LARGER avg weight than HIGH-vol (commodities).
    W_ls = compute_target_weights(composite, prices, mode="long_short", vol_window=WINDOW)
    tbl = pd.DataFrame({"avg_abs_weight": W_ls.abs().mean(), "ann_vol": asset_vol})
    tbl = tbl.sort_values("avg_abs_weight", ascending=False)
    print("=" * 70)
    print("  INVERSE-VOL CHECK (long_short): avg |weight| vs asset vol")
    print("=" * 70)
    print(tbl.to_string())


if __name__ == "__main__":
    main()