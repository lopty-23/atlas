"""Data validation checks for raw price and macro data.

Run right after ingest to catch silently bad data before it propagates
downstream. Each check returns a list of issue strings — empty list means
clean. The caller decides whether to log, warn, or halt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = []
        if self.errors:
            lines.append(f"ERRORS ({len(self.errors)}):")
            lines.extend(f"  - {e}" for e in self.errors)
        if self.warnings:
            lines.append(f"WARNINGS ({len(self.warnings)}):")
            lines.extend(f"  - {w}" for w in self.warnings)
        if not lines:
            lines.append("All checks passed.")
        return "\n".join(lines)


def validate_prices(prices: pd.DataFrame, max_staleness_days: int = 5) -> ValidationReport:
    """Run sanity checks on an ETF price DataFrame."""
    report = ValidationReport()

    if prices.empty:
        report.errors.append("Price frame is empty")
        return report

    # ERROR: any non-positive prices (negative or zero)
    non_positive = (prices <= 0).any()
    for col in non_positive[non_positive].index:
        first_bad = (prices[col] <= 0).idxmax()
        report.errors.append(
            f"{col}: non-positive price (e.g. {prices[col].loc[first_bad]:.2f} on {first_bad.date()})"
        )

    # ERROR: any column with no data at all (delisted or bad ticker)
    all_nan = prices.isna().all()
    for col in all_nan[all_nan].index:
        report.errors.append(f"{col}: no data at all (likely bad ticker)")

    # WARNING: extreme daily returns (>40% in a day → probably unadjusted split)
    returns = prices.pct_change()
    extreme = (returns.abs() > 0.40).any()
    for col in extreme[extreme].index:
        worst_date = returns[col].abs().idxmax()
        worst_value = returns[col].loc[worst_date]
        report.warnings.append(
            f"{col}: extreme return {worst_value:.1%} on {worst_date.date()} "
            f"(may indicate unadjusted split)"
        )

    # WARNING: stale data (last observation older than max_staleness_days)
    last_obs = prices.dropna(how="all").index.max()
    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.tseries.offsets.BDay(max_staleness_days)
    if last_obs < cutoff:
        days_old = (today - last_obs).days
        report.warnings.append(
            f"Data is stale: last observation {last_obs.date()} ({days_old} calendar days ago)"
        )

    # WARNING: columns with high missing-data percentages
    missing_pct = prices.isna().mean()
    high_missing = missing_pct[missing_pct > 0.5]
    for col, pct in high_missing.items():
        report.warnings.append(f"{col}: {pct:.0%} missing values (younger ETF or data gap)")

    return report


def validate_fred(fred: pd.DataFrame) -> ValidationReport:
    """Run sanity checks on FRED macro data.

    FRED data is messier than price data — series have different frequencies,
    long historical gaps are common (e.g. NFCI starts in 1971 but some series
    only started in 2000s), and revisions happen. Be lenient.
    """
    report = ValidationReport()

    if fred.empty:
        report.errors.append("FRED frame is empty")
        return report

    # ERROR: completely missing series
    all_nan = fred.isna().all()
    for col in all_nan[all_nan].index:
        report.errors.append(f"{col}: no data returned from FRED")

    # WARNING: very short series (less than 1 year of data)
    for col in fred.columns:
        non_null_count = fred[col].notna().sum()
        if 0 < non_null_count < 250:  # ~1 year of daily, or 12 months
            report.warnings.append(
                f"{col}: only {non_null_count} observations (very short series)"
            )

    return report