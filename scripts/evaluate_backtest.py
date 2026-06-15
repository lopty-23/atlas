"""Results stage 1 -- beta decomposition vs SPY, plus a 60/40 benchmark.

Run: uv run python scripts/evaluate_backtest.py
Alpha t-stat: mean(excess_p - beta*excess_m) / Newey-West SE (n_lags = one
rebalance cycle), conditioning on the estimated beta.
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


def regress_on_market(excess_p: pd.Series, excess_m: pd.Series) -> dict:
    """OLS of portfolio excess on market excess; HAC t-stat on alpha."""
    df = pd.concat({"p": excess_p, "m": excess_m}, axis=1).dropna()
    beta = df["p"].cov(df["m"]) / df["m"].var()
    alpha_series = df["p"] - beta * df["m"]
    alpha_daily = alpha_series.mean()
    se = _newey_west_se(alpha_series, n_lags=NW_LAGS)
    return {
        "alpha_ann": alpha_daily * TRADING_DAYS,
        "alpha_t": alpha_daily / se,
        "beta": beta,
        "r2": df["p"].corr(df["m"]) ** 2,
    }


def scorecard(label: str, returns: pd.Series, rf: pd.Series, reg: dict) -> None:
    s = summarize(returns, rf)
    print(f"{label:<12} Sharpe {s['sharpe']:.2f}  CAGR {s['cagr']:.2%}  "
          f"Vol {s['ann_vol']:.3f}  MaxDD {s['max_drawdown']:.2%}")
    print(f"{'':<12} alpha {reg['alpha_ann']:+.2%}/y (t={reg['alpha_t']:.2f})  "
          f"beta {reg['beta']:+.2f}  R2 {reg['r2']:.2f}")


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
    rf = macro["DGS3MO"] / 100.0 / TRADING_DAYS
    rf_al = rf.reindex(composite.index).ffill().fillna(0.0)
    excess_m = prices["SPY"].reindex(composite.index).pct_change() - rf_al

    live_idx = None
    for mode in ("long_short", "long_only"):
        sized = compute_target_weights(composite, prices, mode=mode,
                                       target_vol=0.10, vol_window=126)
        capped = apply_risk_limits(sized, cfg["risk_buckets"],
                                   **cfg["risk_limits"])
        bt = run_backtest(capped, prices, rf)
        live = bt["weights_held"].abs().sum(axis=1) > 0
        r_live = bt["returns"][live]
        if live_idx is None:
            live_idx = r_live.index

        reg = regress_on_market(r_live - rf_al[r_live.index],
                                excess_m[r_live.index])
        print("=" * 70)
        scorecard(mode, r_live, rf, reg)

    # 60/40 SPY/IEF through the SAME engine (same costs, lag, cash leg),
    # scored over the same live window.
    targets = pd.DataFrame({"SPY": 0.60, "IEF": 0.40}, index=composite.index)
    bt_b = run_backtest(targets, prices[["SPY", "IEF"]], rf)
    r_b = bt_b["returns"].reindex(live_idx)
    reg_b = regress_on_market(r_b - rf_al[live_idx], excess_m[live_idx])
    print("=" * 70)
    scorecard("60/40", r_b, rf, reg_b)


if __name__ == "__main__":
    main()