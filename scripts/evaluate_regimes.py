"""Results stage 2 -- regime slices + the diversifier (portable-alpha) test.

Run: uv run python scripts/evaluate_regimes.py
Regime windows are set on PUBLIC macro events, fixed before seeing results
(see DECISIONS / chat). Slices are low-N illustrations of behavior, NOT
statistically significant sub-period claims.
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
from atlas.backtest.performance import summarize, max_drawdown

TRADING_DAYS = 252

REGIMES = {
    "GFC          ": ("2007-10-01", "2009-03-31"),
    "QE calm      ": ("2012-01-01", "2019-12-31"),
    "COVID        ": ("2020-02-01", "2020-04-30"),
    "Inflation    ": ("2021-04-01", "2023-07-31"),
}


def slice_metrics(returns: pd.Series, rf: pd.Series, lo: str, hi: str) -> dict:
    """Annualized CAGR/Sharpe/MaxDD on the regime sub-window."""
    win = returns.loc[lo:hi]
    if len(win) < 20:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan, "n": len(win)}
    s = summarize(win, rf)
    return {"cagr": s["cagr"], "sharpe": s["sharpe"],
            "maxdd": s["max_drawdown"], "n": len(win)}


def worst_drawdown_dates(returns: pd.Series) -> tuple:
    """Peak / trough / recovery dates of the deepest drawdown."""
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough = dd.idxmin()
    peak_date = equity.loc[:trough].idxmax()
    after = equity.loc[trough:]
    recovered = after[after >= equity.loc[peak_date]]
    rec_date = recovered.index[0] if len(recovered) else None
    return peak_date, trough, rec_date, dd.min()


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

    mode_returns = {}
    for mode in ("long_short", "long_only"):
        sized = compute_target_weights(composite, prices, mode=mode,
                                       target_vol=0.10, vol_window=126)
        capped = apply_risk_limits(sized, cfg["risk_buckets"], **cfg["risk_limits"])
        bt = run_backtest(capped, prices, rf)
        live = bt["weights_held"].abs().sum(axis=1) > 0
        mode_returns[mode] = bt["returns"][live]

    # --- Regime slices -------------------------------------------------------
    for mode, r in mode_returns.items():
        print("=" * 70)
        print(f"  {mode}")
        print("=" * 70)
        for name, (lo, hi) in REGIMES.items():
            m = slice_metrics(r, rf, lo, hi)
            print(f"{name} CAGR {m['cagr']:+7.2%}  Sharpe {m['sharpe']:+5.2f}  "
                  f"MaxDD {m['maxdd']:+7.2%}  ({m['n']}d)")
        pk, tr, rc, depth = worst_drawdown_dates(r)
        rc_s = rc.date() if rc is not None else "not recovered"
        print(f"Worst DD {depth:.2%}: peak {pk.date()} -> trough {tr.date()} "
              f"-> recovery {rc_s}")
        print()

    # --- Diversifier / portable-alpha test (long_short vs 60/40) -------------
    targets = pd.DataFrame({"SPY": 0.60, "IEF": 0.40}, index=composite.index)
    bt_b = run_backtest(targets, prices[["SPY", "IEF"]], rf)
    ls = mode_returns["long_short"]
    bench = bt_b["returns"].reindex(ls.index)

    corr = ls.corr(bench)
    blend = 0.70 * bench + 0.30 * ls          # 70% 60/40 + 30% long-short
    print("=" * 70)
    print("  DIVERSIFIER TEST: long_short as an overlay on 60/40")
    print("=" * 70)
    print(f"corr(long_short, 60/40)      : {corr:+.2f}")
    print(f"60/40 alone        Sharpe    : {summarize(bench, rf)['sharpe']:.2f}")
    print(f"long_short alone   Sharpe    : {summarize(ls, rf)['sharpe']:.2f}")
    print(f"70/30 blend        Sharpe    : {summarize(blend, rf)['sharpe']:.2f}  "
          f"MaxDD {max_drawdown(blend):.2%}")


if __name__ == "__main__":
    main()