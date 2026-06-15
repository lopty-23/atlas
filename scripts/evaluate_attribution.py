"""Results stage 3 -- attribution: per-signal (leave-one-in) and per-bucket.

Run: uv run python scripts/evaluate_attribution.py
Each signal runs through the FULL stack alone, then regime-sliced, does
NOT sum to the combined book. Plus turnover decomposition (signal vs 
vol-scalar vs cap toggling).
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
from atlas.backtest.performance import summarize

TRADING_DAYS = 252
MODE = "long_short"          # attribution focuses on the crisis-alpha book

REGIMES = {
    "GFC      ": ("2007-10-01", "2009-03-31"),
    "QE calm  ": ("2012-01-01", "2019-12-31"),
    "COVID    ": ("2020-02-01", "2020-04-30"),
    "Inflation": ("2021-04-01", "2023-07-31"),
}


def run_stack(roster: dict, prices, cfg, rf) -> dict:
    """Full blend -> size -> risk -> backtest for a given roster dict."""
    composite = blend_signals(roster, scales=ROSTER_SCALES)
    sized = compute_target_weights(composite, prices, mode=MODE,
                                   target_vol=0.10, vol_window=126)
    capped = apply_risk_limits(sized, cfg["risk_buckets"], **cfg["risk_limits"])
    return run_backtest(capped, prices, rf)


def regime_row(returns: pd.Series, rf: pd.Series) -> str:
    cells = []
    for name, (lo, hi) in REGIMES.items():
        win = returns.loc[lo:hi]
        sh = summarize(win, rf)["sharpe"] if len(win) >= 20 else np.nan
        cells.append(f"{name} {sh:+5.2f}")
    full = summarize(returns[returns.index], rf)["sharpe"]
    return "  ".join(cells) + f"   | full {full:+.2f}"


def main() -> None:
    with open("config/universe.yaml") as f:
        cfg = yaml.safe_load(f)
    prices = pd.read_parquet("data/etf_prices.parquet")
    prices_unadj = pd.read_parquet("data/etf_prices_unadjusted.parquet")
    dividends = pd.read_parquet("data/etf_dividends.parquet")
    fred_raw = pd.read_parquet("data/fred_raw.parquet")
    fred_vintages = pd.read_parquet("data/fred_vintages.parquet")
    macro = build_pit_macro(fred_raw, fred_vintages, prices, cfg)

    full_roster = build_roster(prices=prices, prices_unadjusted=prices_unadj,
                               dividends=dividends, macro=macro)
    rf = macro["DGS3MO"] / 100.0 / TRADING_DAYS

    # --- APPROACH A: per-signal leave-one-in, regime-sliced -----------------
    print("=" * 78)
    print("  PER-SIGNAL (leave-one-in, Sharpe per regime) -- does NOT sum to combined")
    print("=" * 78)
    combined = run_stack(full_roster, prices, cfg, rf)
    c_live = combined["returns"][combined["weights_held"].abs().sum(axis=1) > 0]
    print(f"{'COMBINED':<14}{regime_row(c_live, rf)}")
    print("-" * 78)
    for name in full_roster:
        bt = run_stack({name: full_roster[name]}, prices, cfg, rf)
        live = bt["returns"][bt["weights_held"].abs().sum(axis=1) > 0]
        print(f"{name:<14}{regime_row(live, rf)}")

    # --- APPROACH B: per-bucket contribution to the COMBINED book ------------
    # Daily contribution of bucket b = sum_{i in b} w_held[i] * ret[i].
    # These sum across buckets to the combined book's asset return (ex cash leg),
    # so the decomposition is exact.
    rets = prices.reindex(columns=combined["weights_held"].columns).pct_change()
    wheld = combined["weights_held"]
    contrib = (wheld * rets).fillna(0.0)
    live_mask = wheld.abs().sum(axis=1) > 0

    print("\n" + "=" * 78)
    print("  PER-BUCKET contribution to COMBINED (annualized mean return, full period)")
    print("=" * 78)
    for b, tickers in cfg["risk_buckets"].items():
        cols = [t for t in tickers if t in contrib.columns]
        daily = contrib[cols].sum(axis=1)[live_mask]
        print(f"{b:<14}{daily.mean() * TRADING_DAYS:+7.2%}/y")
    total = contrib.sum(axis=1)[live_mask]
    print(f"{'TOTAL (assets)':<14}{total.mean() * TRADING_DAYS:+7.2%}/y "
          f"(ex cash leg)")

    # --- Turnover decomposition ---------------------------------------------
    # Re-run with leverage cap effectively OFF (max_gross huge) to isolate how
    # much turnover the cap's on/off toggling adds vs the uncapped book.
    print("\n" + "=" * 78)
    print("  TURNOVER DECOMPOSITION (annualized gross traded)")
    print("=" * 78)
    composite = blend_signals(full_roster, scales=ROSTER_SCALES)
    sized = compute_target_weights(composite, prices, mode=MODE,
                                   target_vol=0.10, vol_window=126)
    years = len(c_live) / TRADING_DAYS

    capped = apply_risk_limits(sized, cfg["risk_buckets"], **cfg["risk_limits"])
    to_capped = run_backtest(capped, prices, rf)["turnover"].sum() / years

    limits_nocap = {**cfg["risk_limits"], "max_gross": 1e9}
    uncapped = apply_risk_limits(sized, cfg["risk_buckets"], **limits_nocap)
    to_uncapped = run_backtest(uncapped, prices, rf)["turnover"].sum() / years

    print(f"{'with cap':<20}{to_capped:6.2f}x/y   (production)")
    print(f"{'cap OFF':<20}{to_uncapped:6.2f}x/y   (isolates signal+vol-scalar)")
    print(f"{'cap toggling adds':<20}{to_capped - to_uncapped:+6.2f}x/y")


if __name__ == "__main__":
    main()