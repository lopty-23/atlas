"""Results stage 4 -- sensitivity sweeps (one parameter at a time).

Run: uv run python scripts/evaluate_sensitivity.py
Each parameter swept with the OTHER three at baseline. We do NOT pick a "best"
setting -- the point is whether the baseline verdict (crisis-alpha + diversifier
+ inflation protection) is TYPICAL among neighbors or an outlier. Same 22y
sample throughout: this tests fragility-to-knobs, NOT out-of-sample (that needs
walk-forward, separate exercise).
"""
from __future__ import annotations

import yaml
import numpy as np
import pandas as pd

from atlas.data.point_in_time import build_pit_macro
from atlas.evaluation.ic import _newey_west_se
from atlas.portfolio.roster import build_roster, ROSTER_SCALES
from atlas.portfolio.blend import blend_signals
from atlas.portfolio.sizing import compute_target_weights
from atlas.portfolio.risk import apply_risk_limits
from atlas.backtest.engine import run_backtest
from atlas.backtest.performance import summarize

TRADING_DAYS = 252
NW_LAGS = 21
MODE = "long_short"
INFL = ("2021-04-01", "2023-07-31")

BASE = {"vol_window": 126, "rebalance": "ME", "max_gross": 3.0,
        "cost_per_side": 0.0001}


def metrics(composite, prices, cfg, rf, excess_m, p) -> dict:
    """Run the stack with parameter dict p; return verdict-relevant metrics."""
    sized = compute_target_weights(composite, prices, mode=MODE,
                                   target_vol=0.10, vol_window=p["vol_window"])
    limits = {**cfg["risk_limits"], "max_gross": p["max_gross"]}
    capped = apply_risk_limits(sized, cfg["risk_buckets"], **limits)
    bt = run_backtest(capped, prices, rf, rebalance=p["rebalance"],
                      cost_per_side=p["cost_per_side"])
    live = bt["weights_held"].abs().sum(axis=1) > 0
    r = bt["returns"][live]
    rf_l = rf.reindex(r.index).ffill().fillna(0.0)

    # alpha vs SPY
    df = pd.concat({"p": r - rf_l, "m": excess_m.reindex(r.index)}, axis=1).dropna()
    beta = df["p"].cov(df["m"]) / df["m"].var()
    a = df["p"] - beta * df["m"]
    alpha_ann = a.mean() * TRADING_DAYS
    alpha_t = a.mean() / _newey_west_se(a, n_lags=NW_LAGS)

    # diversifier: 70/30 blend with 60/40
    bench = run_backtest(pd.DataFrame({"SPY": 0.60, "IEF": 0.40}, index=composite.index),
                         prices[["SPY", "IEF"]], rf,
                         rebalance=p["rebalance"], cost_per_side=p["cost_per_side"]
                         )["returns"].reindex(r.index)
    blend_sh = summarize(0.70 * bench + 0.30 * r, rf)["sharpe"]

    infl_sh = summarize(r.loc[INFL[0]:INFL[1]], rf)["sharpe"]
    s = summarize(r, rf)
    return {"sharpe": s["sharpe"], "alpha": alpha_ann, "t": alpha_t,
            "maxdd": s["max_drawdown"], "blend": blend_sh, "infl": infl_sh}


def sweep(label, key, values, composite, prices, cfg, rf, excess_m):
    print("=" * 78)
    print(f"  SWEEP: {label}   (baseline = {BASE[key]})")
    print(f"  {'value':>8} {'Sharpe':>7} {'alpha':>8} {'t':>6} "
          f"{'maxDD':>8} {'70/30':>7} {'infl Sh':>8}")
    print("-" * 78)
    for v in values:
        p = {**BASE, key: v}
        m = metrics(composite, prices, cfg, rf, excess_m, p)
        mark = "  <- base" if v == BASE[key] else ""
        print(f"  {str(v):>8} {m['sharpe']:>7.2f} {m['alpha']:>+8.2%} "
              f"{m['t']:>6.2f} {m['maxdd']:>+8.2%} {m['blend']:>7.2f} "
              f"{m['infl']:>+8.2f}{mark}")
    print()


def main() -> None:
    with open("config/universe.yaml") as f:
        cfg = yaml.safe_load(f)
    prices = pd.read_parquet("data/etf_prices.parquet")
    prices_unadj = pd.read_parquet("data/etf_prices_unadjusted.parquet")
    dividends = pd.read_parquet("data/etf_dividends.parquet")
    fred_raw = pd.read_parquet("data/fred_raw.parquet")
    fred_vintages = pd.read_parquet("data/fred_vintages.parquet")
    macro = build_pit_macro(fred_raw, fred_vintages, prices, cfg)

    composite = blend_signals(
        build_roster(prices=prices, prices_unadjusted=prices_unadj,
                     dividends=dividends, macro=macro),
        scales=ROSTER_SCALES)
    rf = macro["DGS3MO"] / 100.0 / TRADING_DAYS
    excess_m = prices["SPY"].reindex(composite.index).pct_change() \
        - rf.reindex(composite.index).ffill().fillna(0.0)

    sweep("vol window (days)", "vol_window", [63, 126, 252],
          composite, prices, cfg, rf, excess_m)
    sweep("rebalance freq", "rebalance", ["W", "ME", "QE"],
          composite, prices, cfg, rf, excess_m)
    sweep("leverage cap", "max_gross", [2.0, 3.0, 4.0],
          composite, prices, cfg, rf, excess_m)
    sweep("cost per side", "cost_per_side", [0.00005, 0.0001, 0.0002, 0.0005],
          composite, prices, cfg, rf, excess_m)


if __name__ == "__main__":
    main()