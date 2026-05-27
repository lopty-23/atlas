"""Run the full data ingestion pipeline: ETFs + non-vintage FRED + vintages.

This is the entry point you run when you want to refresh your data cache.
It writes three parquet files to data/:
    etf_prices.parquet
    fred_raw.parquet
    fred_vintages.parquet
"""
from __future__ import annotations

from pathlib import Path
import time

from atlas.data.ingest import ingest_all

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "universe.yaml"


def main() -> None:
    print(f"Loading universe from: {CONFIG_PATH}")
    t0 = time.perf_counter()
    ingest_all(CONFIG_PATH)
    elapsed = time.perf_counter() - t0
    print(f"\nIngest complete in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()