"""Results stage 5 -- "where does the strategy bleed" diagnostic (long_short).

Run: uv run python scripts/evaluate_bleed.py

DIAGNOSTIC ONLY: characterizes the weak periods at finer grain than the four
coarse regime windows. It does NOT fit a regime detector to them (the curve-fit
trap). Everything here is in-sample description that checks consistency with the
standing "trendless calm" story; it cannot certify that story out of sample --
that is the walk-forward gap.

Bleed is ABSOLUTE / excess-over-rf, never benchmark-relative: the book is
near-market-neutral (beta ~0.10), so trailing SPY in a bull market is the hedge
working, not a bleed.

Signal-state note: TSMomentum/BondCarry/FXCarry are per-date cross-sectionally
z-scored, so their cross-sectional dispersion is ~1 every day BY CONSTRUCTION
and carries no regime information -- "momentum quiet" is read off CONTRIBUTION
(layer 3), not magnitude. InflationTrend is time-series normalized, so its
magnitude is meaningful and is reported directly.
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
MODE = "long_short"                 # the crisis-alpha book; its calm-weakness is the subject
ROLL_WINDOWS = (126, 252, 504)
PRIMARY_WINDOW = 252
N_DRAWDOWNS = 8                     # episodes listed in the WHEN table
N_EPISODES_DETAIL = 3              # episodes given the full layer-3 breakdown
QE_CALM = ("2012-01-01", "2019-12-31")


def run_stack(roster: dict, prices: pd.DataFrame, cfg: dict,
              rf: pd.Series) -> dict:
    """Full blend -> size -> risk -> backtest for a roster dict (long_short)."""
    composite = blend_signals(roster, scales=ROSTER_SCALES)
    sized = compute_target_weights(composite, prices, mode=MODE,
                                   target_vol=0.10, vol_window=126)
    capped = apply_risk_limits(sized, cfg["risk_buckets"], **cfg["risk_limits"])
    return run_backtest(capped, prices, rf)


def live_returns(bt: dict) -> pd.Series:
    """Engine returns restricted to days the book actually holds risk."""
    live = bt["weights_held"].abs().sum(axis=1) > 0
    return bt["returns"][live]


def rolling_badness(excess: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Trailing annualized excess return (arithmetic) and Sharpe."""
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std()
    return mean * TRADING_DAYS, mean / std * np.sqrt(TRADING_DAYS)


def drawdown_episodes(returns: pd.Series) -> list[dict]:
    """All drawdown episodes (peak -> trough -> recovery), worst-first.

    Episodes are the contiguous underwater stretches of the equity curve, so
    they are non-overlapping by construction. recovery=None if the episode has
    not recovered by the end of the sample.
    """
    equity = (1.0 + returns).cumprod()
    dates = equity.index
    eq = equity.to_numpy()
    pk = equity.cummax().to_numpy()
    under = eq < pk

    episodes: list[dict] = []
    i, n = 0, len(eq)
    while i < n:
        if not under[i]:
            i += 1
            continue
        start = i                               # first underwater day
        while i < n and under[i]:
            i += 1
        end = i - 1                             # last underwater day; i is recovery (or n)
        peak_val = pk[start]                    # the high that was breached
        peak_idx = int(np.argmax(eq[:start]))   # start >= 1 always (eq[0] is its own peak)
        t_off = start + int(np.argmin(eq[start:end + 1]))
        rec_idx = i if i < n else n - 1
        episodes.append({
            "peak": dates[peak_idx],
            "trough": dates[t_off],
            "recovery": dates[i] if i < n else None,
            "depth": eq[t_off] / peak_val - 1.0,
            "peak_to_trough_days": t_off - peak_idx,
            "underwater_days": rec_idx - peak_idx,
        })
    episodes.sort(key=lambda e: e["depth"])
    return episodes


def concentration(returns: pd.Series, rf: pd.Series, lo: str, hi: str) -> dict:
    """How concentrated are the losses in [lo, hi]? Monthly-excess based.

    worst_quartile_share = (sum of the worst 25% of monthly excess) /
    (sum of all negative monthly excess). ~1.0 means a few months own the
    damage (concentrated); ~0.25-0.50 means losses are spread (uniform grind).
    """
    r = returns.loc[lo:hi]
    rf_w = rf.reindex(r.index).ffill().fillna(0.0)
    m_exc = ((1.0 + r).resample("ME").prod() - 1.0
             - rf_w.resample("ME").sum()).dropna()
    n = len(m_exc)
    neg = m_exc[m_exc < 0.0]
    total_neg = neg.sum()
    k = max(1, int(np.ceil(n * 0.25)))
    share = np.nan if np.isclose(total_neg, 0.0) else m_exc.nsmallest(k).sum() / total_neg
    return {"months": n, "pct_negative": len(neg) / n if n else np.nan,
            "mean_monthly_excess": m_exc.mean(), "worst_quartile_share": share}


def bucket_contrib(wheld: pd.DataFrame, prices: pd.DataFrame, cfg: dict,
                   lo: str, hi: str) -> dict:
    """Annualized per-bucket contribution within [lo, hi]. Exact: the buckets
    sum to the book's asset return ex the cash leg."""
    rets = prices.reindex(columns=wheld.columns).pct_change()
    contrib = (wheld * rets).fillna(0.0).loc[lo:hi]
    out = {b: contrib[[t for t in tickers if t in contrib.columns]].sum(axis=1).mean()
              * TRADING_DAYS
           for b, tickers in cfg["risk_buckets"].items()}
    out["TOTAL"] = contrib.sum(axis=1).mean() * TRADING_DAYS
    return out


def signal_sharpe(signal_rets: dict, rf: pd.Series, lo: str, hi: str) -> dict:
    """Leave-one-in Sharpe per signal within [lo, hi] (interpretive; does NOT
    sum to the combined book)."""
    out = {}
    for name, r in signal_rets.items():
        win = r.loc[lo:hi]
        out[name] = summarize(win, rf)["sharpe"] if len(win) >= 20 else np.nan
    return out


def inflation_magnitude(signals: dict, live_index: pd.DatetimeIndex,
                        lo: str, hi: str) -> dict:
    """Mean |InflationTrend| in [lo, hi] vs its full live-sample average. <1
    means the signal was quieter than usual (its blend voice shrinks when
    small, by the ROSTER_SCALES design)."""
    m = signals["InflationTrend"].abs().mean(axis=1)
    full, win = m.reindex(live_index).mean(), m.loc[lo:hi].mean()
    return {"window": win, "full": full,
            "ratio": np.nan if np.isclose(full, 0.0) else win / full}


def gross_state(wheld: pd.DataFrame, live_index: pd.DatetimeIndex,
                lo: str, hi: str) -> dict:
    """Mean gross leverage (sum|w|) in [lo, hi] vs full live-sample. The
    vol-target scalar levers UP in low realized vol, so a calm bleed can be a
    high-gross grind."""
    gross = wheld.abs().sum(axis=1)
    return {"window": gross.loc[lo:hi].mean(), "full": gross.reindex(live_index).mean()}


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
    rf = macro["DGS3MO"] / 100.0 / TRADING_DAYS

    bt = run_stack(signals, prices, cfg, rf)
    r = live_returns(bt)
    wheld = bt["weights_held"]
    excess = r - rf.reindex(r.index).ffill().fillna(0.0)

    # Per-signal standalone books -- run ONCE each, then sliced to every window.
    signal_rets = {name: live_returns(run_stack({name: signals[name]}, prices, cfg, rf))
                   for name in signals}

    # ---- LAYER 1a: WHEN -- rolling badness ---------------------------------
    print("=" * 76)
    print("  LAYER 1a  ROLLING BADNESS  (excess; return = arithmetic-annualized mean)")
    print("=" * 76)
    print(f"{'window':>7}  {'worst ann.exc':>14}  {'(end)':>11}   "
          f"{'worst Sharpe':>12}  {'(end)':>11}   {'days exc<0':>10}")
    for w in ROLL_WINDOWS:
        ann_ret, shp = rolling_badness(excess, w)
        frac_neg = (ann_ret.dropna() < 0).mean()
        print(f"{w:>7}  {ann_ret.min():>+13.2%}  {str(ann_ret.idxmin().date()):>11}   "
              f"{shp.min():>+12.2f}  {str(shp.idxmin().date()):>11}   {frac_neg:>9.1%}")

    # ---- LAYER 1b: drawdown episodes ---------------------------------------
    eps = drawdown_episodes(r)
    print("\n" + "=" * 76)
    print(f"  LAYER 1b  WORST {N_DRAWDOWNS} DRAWDOWN EPISODES  (peak -> trough -> recovery)")
    print("=" * 76)
    print(f"{'depth':>8}  {'peak':>11}  {'trough':>11}  {'recovery':>11}  "
          f"{'pk->tr':>7}  {'u/w':>6}")
    for e in eps[:N_DRAWDOWNS]:
        rec = str(e["recovery"].date()) if e["recovery"] is not None else "(ongoing)"
        print(f"{e['depth']:>+8.2%}  {str(e['peak'].date()):>11}  {str(e['trough'].date()):>11}  "
              f"{rec:>11}  {e['peak_to_trough_days']:>5}d  {e['underwater_days']:>4}d")

    # ---- LAYER 2: SHAPE -- uniform vs concentrated (QE-calm) ---------------
    c = concentration(r, rf, *QE_CALM)
    print("\n" + "=" * 76)
    print(f"  LAYER 2  QE-CALM SHAPE  [{QE_CALM[0]} .. {QE_CALM[1]}]")
    print("=" * 76)
    print(f"months {c['months']}    negative {c['pct_negative']:.0%}    "
          f"mean monthly excess {c['mean_monthly_excess']:+.3%}")
    print(f"worst-quartile share of losses  {c['worst_quartile_share']:.2f}   "
          f"(->1.0 concentrated, ->0.25-0.50 uniform grind)")

    # ---- LAYER 3: WHAT -- per-bucket / per-signal in the worst windows -----
    windows = [(f"DD#{i + 1}  {e['peak'].date()} -> {e['trough'].date()}",
                str(e["peak"].date()), str(e["trough"].date()))
               for i, e in enumerate(eps[:N_EPISODES_DETAIL])]
    windows.append((f"QE-calm  {QE_CALM[0]} .. {QE_CALM[1]}", *QE_CALM))

    for label, lo, hi in windows:
        print("\n" + "=" * 76)
        print(f"  LAYER 3  {label}")
        print("=" * 76)
        bc = bucket_contrib(wheld, prices, cfg, lo, hi)
        print("  per-bucket contribution (annualized, exact):")
        for v, b in sorted((v, b) for b, v in bc.items() if b != "TOTAL"):
            print(f"    {b:<12}{v:>+8.2%}/y")
        print(f"    {'TOTAL':<12}{bc['TOTAL']:>+8.2%}/y  (ex cash leg)")
        ss = signal_sharpe(signal_rets, rf, lo, hi)
        print("  per-signal leave-one-in Sharpe (interpretive):")
        print("    " + "   ".join(f"{k} {v:+.2f}" for k, v in ss.items()))
        im = inflation_magnitude(signals, r.index, lo, hi)
        gs = gross_state(wheld, r.index, lo, hi)
        print(f"  InflationTrend |signal| {im['window']:.2f} vs {im['full']:.2f} full "
              f"(ratio {im['ratio']:.2f})    gross {gs['window']:.2f}x vs {gs['full']:.2f}x full")


if __name__ == "__main__":
    main()