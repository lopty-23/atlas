"""Pull raw price and macro data from Yahoo Finance and FRED.

Public entry points:
    fetch_etf_prices(tickers, start, end) -> DataFrame
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
    """Load the universe YAML config from disk."""
    with open(config_path) as f:
        return yaml.safe_load(f)
    
def fetch_etf_prices(
    tickers: list[str],
    start: str = "2003-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Pull total-return-adjusted close prices for ETFs from Yahoo Finance.

    Uses auto_adjust=True so prices include split + dividend adjustments
    (i.e. these are total-return-equivalent prices for backtesting).

    Returns
    -------
    DataFrame indexed by business-day DatetimeIndex (tz-naive), one column per ticker.
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    # Modern yfinance with group_by="ticker" returns a MultiIndex column frame
    # like ('SPY', 'Close') even for a single ticker. Normalize to a flat
    # per-ticker DataFrame of Close prices.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw.xs("Close", axis=1, level=1)
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(axis=1, how="all")

    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    return prices.sort_index()


def fetch_fred_series(series_ids: list[str]) -> pd.DataFrame:
    """Pull macro time series from FRED.

    Series are indexed by their REFERENCE date (the period the data describes),
    NOT their publication date. Publication-lag handling happens later in
    point_in_time.py.

    Raises
    ------
    RuntimeError if FRED_API_KEY is not in the environment.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY missing. Check that .env exists and contains the key."
        )

    fred = Fred(api_key=api_key)
    frames: dict[str, pd.Series] = {}
    for sid in series_ids:
        s = fred.get_series(sid)
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        frames[sid] = s

    return pd.DataFrame(frames).sort_index()


def cache_path(name: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Return the parquet path for a cached dataset (e.g. 'etf_prices')."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}.parquet"


def ingest_all(config_path: Path) -> None:
    """Top-level orchestrator: read config, fetch all data, write parquet cache.

    This is the only function in this module that performs side effects on
    disk. Everything else returns DataFrames so it's easy to test.
    """
    universe = load_universe(config_path)

    # Flatten the nested ETF config into a single ticker list
    etf_tickers = [
        item["ticker"]
        for asset_class in universe["etfs"].values()
        for item in asset_class
    ]
    print(f"Fetching {len(etf_tickers)} ETFs from Yahoo Finance...")
    prices = fetch_etf_prices(
        etf_tickers,
        start=universe["settings"]["start_date"],
        end=universe["settings"]["end_date"],
    )
    prices.to_parquet(cache_path("etf_prices"))
    print(f"  -> wrote {prices.shape[0]} days x {prices.shape[1]} ETFs")

    # Same flattening for FRED series
    fred_ids = [
        item["id"]
        for group in universe["fred_series"].values()
        for item in group
    ]
    print(f"Fetching {len(fred_ids)} FRED series...")
    fred = fetch_fred_series(fred_ids)
    fred.to_parquet(cache_path("fred_raw"))
    print(f"  -> wrote {fred.shape[0]} rows x {fred.shape[1]} series")