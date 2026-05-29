"""Pull raw price and macro data from Yahoo Finance and FRED.

Public entry points:
    fetch_etf_prices(tickers, start, end) -> DataFrame
    fetch_etf_prices_unadjusted(tickers, start, end) -> DataFrame
    fetch_etf_dividends(tickers, start, end) -> DataFrame
    fetch_fred_series(series_ids) -> DataFrame
    ingest_all(config_path) -> None       # orchestrator: reads config, writes cache
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data"


def load_universe(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_etf_prices(
    tickers: list[str],
    start: str = "2003-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw.xs("Close", axis=1, level=1)
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(axis=1, how="all")

    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    return prices.sort_index()


def fetch_etf_prices_unadjusted(
    tickers: list[str],
    start: str = "2003-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Pull UNADJUSTED closing prices to calculate dividend yields"""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw.xs("Close", axis=1, level=1)
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(axis=1, how="all")
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    return prices.sort_index()


def fetch_etf_dividends(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Pull per-ticker dividend history as a wide DataFrame.yfinance returns 
    dividend timestamps tz-aware at the 09:30 NY open; we strip tz and normalize 
    so dates align with the daily price index (which is midnight tz-naive). 
    """
    series_by_ticker: dict[str, pd.Series] = {}
    for t in tickers:
        s = yf.Ticker(t).dividends.copy()
        if s.empty:
            continue
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="last")]
        if start is not None:
            s = s[s.index >= pd.Timestamp(start)]
        if end is not None:
            s = s[s.index <= pd.Timestamp(end)]
        series_by_ticker[t] = s

    if not series_by_ticker:
        return pd.DataFrame()

    return pd.DataFrame(series_by_ticker).sort_index()


def _get_fred_client() -> Fred:
    """Build a FRED client, raising a clear error if the key is missing."""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY missing. Check that .env exists and contains the key."
        )
    return Fred(api_key=api_key)


def fetch_fred_series(series_ids: list[str]) -> pd.DataFrame:
    """Pull LATEST values of macro time series from FRED for series where
    revisions don't materially matter (fixed-lag).

    Series are indexed by their REFERENCE date (the period the data describes).
    Publication-lag handling happens in point_in_time.py.
    """
    fred = _get_fred_client()
    frames: dict[str, pd.Series] = {}
    for sid in series_ids:
        s = fred.get_series(sid)
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        frames[sid] = s
    return pd.DataFrame(frames).sort_index()


def fetch_fred_vintages(series_ids: list[str]) -> pd.DataFrame:
    """Pull ALL releases of macro series from ALFRED (vintage data).

    Returns the long-format vintage history: every observation period has one
    row per release, with `realtime_start` recording when that value became
    publicly known. 
    """
    fred = _get_fred_client()
    frames: list[pd.DataFrame] = []
    for sid in series_ids:
        df = fred.get_series_all_releases(sid)
        df = df.rename(
            columns={"realtime_start": "vintage_date", "date": "observation_date"}
        )
        df["series_id"] = sid
        df["vintage_date"] = pd.to_datetime(df["vintage_date"])
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        frames.append(df[["series_id", "observation_date", "vintage_date", "value"]])
    return pd.concat(frames, ignore_index=True)


def cache_path(name: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Return the parquet path for a cached dataset (e.g. 'etf_prices')."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}.parquet"


def _split_series_by_vintage_flag(
    universe: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Partition FRED series into (non-vintage, vintage) lists based on the
    `vintage: true` flag in the config."""
    non_vintage: list[str] = []
    vintage: list[str] = []
    for group in universe["fred_series"].values():
        for item in group:
            if item.get("vintage", False):
                vintage.append(item["id"])
            else:
                non_vintage.append(item["id"])
    return non_vintage, vintage


def ingest_all(config_path: Path) -> None:
    """Top-level orchestrator: read config, fetch all data, write parquet cache.

    Writes five parquet files to `data/`:
        etf_prices.parquet              : wide ETF prices, AUTO-ADJUSTED (for returns)
        etf_prices_unadjusted.parquet   : wide ETF prices, UNADJUSTED (for yield denom)
        etf_dividends.parquet           : wide dividend payments, NaN where no payment
        fred_raw.parquet                : wide non-vintage FRED series
        fred_vintages.parquet           : long-format vintage data
    """
    universe = load_universe(config_path)

    # --- ETFs ---
    etf_tickers = [
        item["ticker"]
        for asset_class in universe["etfs"].values()
        for item in asset_class
    ]
    start = universe["settings"]["start_date"]
    end = universe["settings"]["end_date"]

    print(f"Fetching {len(etf_tickers)} ETFs (adjusted close) from Yahoo Finance...")
    prices = fetch_etf_prices(etf_tickers, start=start, end=end)
    prices.to_parquet(cache_path("etf_prices"))
    print(f"  -> wrote {prices.shape[0]} days x {prices.shape[1]} ETFs")

    print(f"Fetching {len(etf_tickers)} ETFs (UNADJUSTED close) from Yahoo Finance...")
    prices_unadj = fetch_etf_prices_unadjusted(etf_tickers, start=start, end=end)
    prices_unadj.to_parquet(cache_path("etf_prices_unadjusted"))
    print(f"  -> wrote {prices_unadj.shape[0]} days x {prices_unadj.shape[1]} ETFs")

    print(f"Fetching dividend history for {len(etf_tickers)} ETFs...")
    divs = fetch_etf_dividends(etf_tickers, start=start, end=end)
    divs.to_parquet(cache_path("etf_dividends"))
    n_payments = int(divs.notna().sum().sum())
    print(f"  -> wrote {divs.shape[0]} ex-dates x {divs.shape[1]} ETFs "
          f"({n_payments} total payments)")

    # --- FRED: split into vintage and non-vintage paths ---
    non_vintage_ids, vintage_ids = _split_series_by_vintage_flag(universe)

    print(f"Fetching {len(non_vintage_ids)} non-vintage FRED series...")
    fred_raw = fetch_fred_series(non_vintage_ids)
    fred_raw.to_parquet(cache_path("fred_raw"))
    print(f"  -> wrote {fred_raw.shape[0]} rows x {fred_raw.shape[1]} series")

    print(
        f"Fetching {len(vintage_ids)} vintage-aware FRED series "
        f"(this is slower; each series has hundreds of releases)..."
    )
    fred_vintages = fetch_fred_vintages(vintage_ids)
    fred_vintages.to_parquet(cache_path("fred_vintages"))
    n_unique_series = fred_vintages["series_id"].nunique()
    print(f"  -> wrote {len(fred_vintages)} rows across {n_unique_series} series")