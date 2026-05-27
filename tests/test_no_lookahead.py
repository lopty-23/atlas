"""Tests proving the data layer doesn't leak future information.

We test both the fixed-lag path (apply_publication_lags) and the vintage
path (align_vintages_to_calendar), with explicit hand-crafted inputs and
known correct outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas.data.point_in_time import (
    align_vintages_to_calendar,
    apply_publication_lags,
    build_pit_macro,
    lag_map_from_config,
    vintage_series_from_config,
)


# ---------------------------------------------------------------------------
# Fixed-lag path
# ---------------------------------------------------------------------------


class TestApplyPublicationLags:
    """Tests for the fixed-lag publication-shift function."""

    def test_value_not_visible_before_publication_date(self) -> None:
        """A value dated Jan 1 with lag=14 must not appear before its publication date."""
        raw = pd.DataFrame(
            {"CPI": [100.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )
        target = pd.bdate_range("2024-01-01", "2024-01-31")
        result = apply_publication_lags(raw, {"CPI": 14}, target)

        # Compute the publication date using the same BDay arithmetic the
        # function uses, so the test can't be wrong about calendar math.
        pub_date = pd.Timestamp("2024-01-01") + pd.tseries.offsets.BDay(14)
        before_pub = result.loc[result.index < pub_date, "CPI"]
        assert before_pub.isna().all(), (
            "Value leaked before its publication date — look-ahead bias present"
        )

    def test_value_visible_on_and_after_publication_date(self) -> None:
        """The same value should appear on its publication date and after."""
        raw = pd.DataFrame(
            {"CPI": [100.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )
        target = pd.bdate_range("2024-01-01", "2024-02-28")
        result = apply_publication_lags(raw, {"CPI": 14}, target)

        # Compute the expected publication date: Jan 1 + 14 business days
        pub_date = pd.Timestamp("2024-01-01") + pd.tseries.offsets.BDay(14)
        on_or_after = result.loc[result.index >= pub_date, "CPI"]
        assert (on_or_after == 100.0).all(), (
            "Value should be visible from publication date onward"
        )

    def test_forward_fill_holds_value_between_publications(self) -> None:
        """Between publications, the latest known value should persist."""
        raw = pd.DataFrame(
            {"CPI": [100.0, 101.0]},
            index=pd.to_datetime(["2024-01-01", "2024-02-01"]),
        )
        target = pd.bdate_range("2024-01-01", "2024-03-15")
        result = apply_publication_lags(raw, {"CPI": 14}, target)

        # After Jan publication, before Feb publication: should hold at 100.0
        first_pub = pd.Timestamp("2024-01-01") + pd.tseries.offsets.BDay(14)
        second_pub = pd.Timestamp("2024-02-01") + pd.tseries.offsets.BDay(14)
        in_between = result.loc[
            (result.index >= first_pub) & (result.index < second_pub),
            "CPI",
        ]
        assert (in_between == 100.0).all(), (
            "Value should hold constant between publications"
        )

        # On/after second publication: should jump to 101.0
        after_second = result.loc[result.index >= second_pub, "CPI"]
        assert (after_second == 101.0).all()

    def test_lag_one_for_daily_series(self) -> None:
        """A daily series with lag=1 should be visible one business day later."""
        raw = pd.DataFrame(
            {"DGS10": [4.5, 4.6, 4.7]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )
        target = pd.bdate_range("2024-01-01", "2024-01-10")
        result = apply_publication_lags(raw, {"DGS10": 1}, target)

        # On Jan 2 (the reference date), value should NOT yet be visible
        assert pd.isna(result.loc["2024-01-02", "DGS10"]), (
            "Value visible on its own reference date — lag=1 not enforced"
        )

        # On Jan 3 (one business day later), the Jan 2 value should appear
        assert result.loc["2024-01-03", "DGS10"] == 4.5

        # On Jan 4: Jan 3 value should be there
        assert result.loc["2024-01-04", "DGS10"] == 4.6

    def test_missing_series_silently_skipped(self) -> None:
        """A series in lag_map but not in raw data should be silently dropped."""
        raw = pd.DataFrame(
            {"CPI": [100.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )
        target = pd.bdate_range("2024-01-01", "2024-02-28")
        result = apply_publication_lags(
            raw, {"CPI": 14, "MISSING_SERIES": 5}, target
        )

        # CPI column should exist; missing series should not be in output
        assert "CPI" in result.columns
        assert "MISSING_SERIES" not in result.columns

    def test_empty_raw_data_produces_empty_columns(self) -> None:
        """If a series has no observations, the function should not crash."""
        raw = pd.DataFrame(
            {"CPI": [np.nan, np.nan]},
            index=pd.to_datetime(["2024-01-01", "2024-02-01"]),
        )
        target = pd.bdate_range("2024-01-01", "2024-02-28")
        # Should not raise — empty series is silently skipped
        result = apply_publication_lags(raw, {"CPI": 14}, target)
        assert "CPI" not in result.columns


# ---------------------------------------------------------------------------
# Vintage path
# ---------------------------------------------------------------------------


class TestAlignVintagesToCalendar:
    """Tests for the ALFRED-vintage alignment function."""

    @staticmethod
    def _make_vintages(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
        """Build a synthetic vintages DataFrame from a list of tuples."""
        df = pd.DataFrame(
            rows,
            columns=["series_id", "observation_date", "vintage_date", "value"],
        )
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df["vintage_date"] = pd.to_datetime(df["vintage_date"])
        return df

    def test_value_not_visible_before_first_vintage(self) -> None:
        """A value cannot appear before its first vintage_date."""
        vintages = self._make_vintages(
            [
                ("GDP", "2024-01-01", "2024-04-25", 1.6),
            ]
        )
        target = pd.bdate_range("2024-01-01", "2024-05-31")
        result = align_vintages_to_calendar(vintages, target)

        before_vintage = result.loc[result.index < "2024-04-25", "GDP"]
        assert before_vintage.isna().all(), (
            "GDP value visible before its first publication — leakage"
        )

    def test_revision_overwrites_earlier_vintage(self) -> None:
        """When a revision comes out, it should overwrite the prior value."""
        vintages = self._make_vintages(
            [
                ("GDP", "2024-01-01", "2024-04-25", 1.6),  # advance
                ("GDP", "2024-01-01", "2024-05-30", 1.3),  # second estimate (revision)
                ("GDP", "2024-01-01", "2024-06-27", 1.4),  # third estimate
            ]
        )
        target = pd.bdate_range("2024-04-01", "2024-07-15")
        result = align_vintages_to_calendar(vintages, target)

        # Use .asof() to look up "the value on or before this date" — robust
        # to whether the queried date happens to be a weekday or weekend.
        assert result["GDP"].asof("2024-05-15") == 1.6  # after 1st, before 2nd
        assert result["GDP"].asof("2024-06-14") == 1.3  # after 2nd, before 3rd
        assert result["GDP"].asof("2024-07-10") == 1.4  # after 3rd

    def test_new_period_overwrites_prior_period_via_ffill(self) -> None:
        """When a new observation period is released, it should become the
        latest known value via forward-fill."""
        vintages = self._make_vintages(
            [
                ("GDP", "2024-01-01", "2024-04-25", 1.6),  # Q1 advance
                ("GDP", "2024-04-01", "2024-07-25", 2.8),  # Q2 advance
            ]
        )
        target = pd.bdate_range("2024-04-01", "2024-08-15")
        result = align_vintages_to_calendar(vintages, target)

        # Between Q1 release and Q2 release: Q1 value
        assert result["GDP"].asof("2024-06-14") == 1.6
        # After Q2 release: Q2 value
        assert result["GDP"].asof("2024-08-01") == 2.8

    def test_multiple_series_handled_independently(self) -> None:
        """Two series should be aligned without interfering with each other."""
        vintages = self._make_vintages(
            [
                ("GDP", "2024-01-01", "2024-04-25", 1.6),
                ("CPI", "2024-01-01", "2024-02-13", 308.0),
                ("CPI", "2024-02-01", "2024-03-12", 309.0),
            ]
        )
        target = pd.bdate_range("2024-01-01", "2024-05-15")
        result = align_vintages_to_calendar(vintages, target)

       # On March 1: CPI knows Jan value (308.0), GDP unknown
        assert result["CPI"].asof("2024-03-01") == 308.0
        assert pd.isna(result["GDP"].asof("2024-03-01"))

        # On May 1: CPI knows Feb value (309.0), GDP knows Q1 (1.6)
        assert result["CPI"].asof("2024-05-01") == 309.0
        assert result["GDP"].asof("2024-05-01") == 1.6

# ---------------------------------------------------------------------------
# Build-pit-macro integration
# ---------------------------------------------------------------------------


class TestBuildPitMacro:
    """Integration tests combining fixed-lag and vintage paths."""

    def test_raises_on_non_datetime_index(self) -> None:
        """Prices with a non-DatetimeIndex should raise TypeError immediately."""
        prices_bad = pd.DataFrame(
            {"SPY": [100.0, 101.0, 102.0]},
            index=[0, 1, 2],  # RangeIndex, not DatetimeIndex
        )
        with pytest.raises(TypeError, match="DatetimeIndex"):
            build_pit_macro(
                raw_fred=pd.DataFrame(),
                vintages=pd.DataFrame(
                    columns=["series_id", "observation_date", "vintage_date", "value"]
                ),
                prices=prices_bad,
                universe_config={"fred_series": {}},
            )

    def test_raises_on_duplicate_series_in_both_paths(self) -> None:
        """If a series ID appears in both vintage and non-vintage configs,
        the function should detect the conflict and raise."""
        prices = pd.DataFrame(
            {"SPY": [100.0]},
            index=pd.bdate_range("2024-01-01", periods=1),
        )
        # Config has CPI in non-vintage path
        config = {
            "fred_series": {
                "inflation": [
                    {"id": "CPI", "lag_days": 14},  # no vintage flag
                ]
            }
        }
        raw_fred = pd.DataFrame(
            {"CPI": [100.0]}, index=pd.to_datetime(["2023-12-01"])
        )
        # ALSO has CPI in vintage data — this is the bug case
        vintages = pd.DataFrame(
            {
                "series_id": ["CPI"],
                "observation_date": pd.to_datetime(["2023-12-01"]),
                "vintage_date": pd.to_datetime(["2023-12-15"]),
                "value": [100.0],
            }
        )
        with pytest.raises(ValueError, match="both"):
            build_pit_macro(raw_fred, vintages, prices, config)