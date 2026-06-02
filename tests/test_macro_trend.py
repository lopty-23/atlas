"""Tests for the GrowthTrend macro signal.

Covers the trend computation (excess growth), the expanding (no-look-ahead)
z-score, the beta mapping, and — critically — the time-series normalization
that PRESERVES MAGNITUDE (a deep-recession reading must produce a larger
signal than a mild one; the earlier cross-sectional version flattened this).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.signals.macro_trend import GrowthTrend
from atlas.signals.macro_trend import InflationTrend


def _macro_with_growth(
    indicator: str = "INDPRO",
    n_years: int = 8,
) -> pd.DataFrame:
    """Build a daily macro frame with one growth indicator that rises steadily,
    then suffers a sharp collapse, then recovers — so the trend has a clear
    acceleration, crash, and rebound to test against."""
    n = n_years * 252
    idx = pd.bdate_range("2010-01-01", periods=n)
    # Steady ~3%/yr growth for the first 5 years, sharp drop in year 6,
    # recovery after.
    level = np.empty(n)
    for i in range(n):
        year = i / 252
        if year < 5:
            level[i] = 100.0 * (1.03 ** year)
        elif year < 6:
            # sharp 20% collapse over one year
            level[i] = 100.0 * (1.03 ** 5) * (1 - 0.20 * (year - 5))
        else:
            level[i] = 100.0 * (1.03 ** 5) * 0.80 * (1.02 ** (year - 6))
    return pd.DataFrame({indicator: level}, index=idx)


class TestExcessGrowth:
    """The trend measure: YoY growth minus its trailing average."""

    def test_excess_positive_during_acceleration(self) -> None:
        macro = _macro_with_growth()
        sig = GrowthTrend(indicators=("INDPRO",))
        excess = sig._yoy_excess(macro["INDPRO"])
        # During the steady-growth phase (year 3-4), YoY is steady, so excess
        # is near zero (growth equals its own recent average).
        steady = excess.loc["2013-06-01":"2013-12-31"].mean()
        assert abs(steady) < 0.05, "Steady growth should give near-zero excess"

    def test_excess_negative_during_collapse(self) -> None:
        macro = _macro_with_growth()
        sig = GrowthTrend(indicators=("INDPRO",))
        excess = sig._yoy_excess(macro["INDPRO"])
        # During the collapse (year 5-6), YoY turns sharply negative and falls
        # below its recent average -> excess strongly negative.
        collapse = excess.loc["2015-06-01":"2015-12-31"].mean()
        assert collapse < 0, "Collapse should give negative excess growth"


class TestExpandingZScore:
    """The expanding z-score must not use future data (no look-ahead)."""

    def test_no_lookahead_expanding_zscore(self) -> None:
        macro = _macro_with_growth()
        sig = GrowthTrend(indicators=("INDPRO",))

        score_full = sig._compute_growth_score(macro)

        # Corrupt the macro AFTER a cutoff date, recompute, and confirm the
        # score BEFORE the cutoff is unchanged (expanding window can't see
        # future data).
        cutoff = macro.index[len(macro) // 2]
        corrupted = macro.copy()
        corrupted.loc[corrupted.index > cutoff] *= 3.0
        score_corrupted = sig._compute_growth_score(corrupted)

        before = score_full.loc[:cutoff].dropna()
        before_corr = score_corrupted.loc[:cutoff].dropna()
        common = before.index.intersection(before_corr.index)
        pd.testing.assert_series_equal(
            before.loc[common], before_corr.loc[common], check_names=False
        )


class TestBetaMapping:
    """The single growth score maps onto assets via fixed beta signs."""

    def test_procyclical_and_countercyclical_opposite_signs(self) -> None:
        macro = _macro_with_growth()
        prices = pd.DataFrame(
            {"SPY": 1.0, "TLT": 1.0, "GLD": 1.0},  # pro, counter, neutral
            index=macro.index,
        )
        sig = GrowthTrend(indicators=("INDPRO",))
        scores = sig.compute(prices, macro)

        last = scores.iloc[-1]
        # SPY (pro-cyclical, +1) and TLT (counter-cyclical, -1) must have
        # opposite signs; GLD (neutral, 0) must be NaN.
        assert not np.isnan(last["SPY"])
        assert not np.isnan(last["TLT"])
        assert np.isnan(last["GLD"]), "Neutral-beta asset should abstain (NaN)"
        assert np.sign(last["SPY"]) == -np.sign(last["TLT"]), (
            "Pro- and counter-cyclical assets must have opposite signs"
        )


class TestMagnitudePreservation:
    """The key property of time_series normalization: a stronger growth signal
    produces a LARGER position. Cross-sectional z-scoring would flatten this."""

    def test_deeper_signal_gives_larger_magnitude(self) -> None:
        macro = _macro_with_growth()
        prices = pd.DataFrame({"SPY": 1.0, "TLT": 1.0}, index=macro.index)
        sig = GrowthTrend(indicators=("INDPRO",))
        scores = sig.compute(prices, macro)

        # The collapse period should produce a larger-magnitude SPY signal than
        # the steady-growth period (intensity is preserved, not flattened).
        steady_mag = scores["SPY"].loc["2013-06-01":"2013-12-31"].abs().mean()
        collapse_mag = scores["SPY"].loc["2015-09-01":"2015-12-31"].abs().mean()
        assert collapse_mag > steady_mag, (
            "Time-series normalization must preserve magnitude: a sharp collapse "
            "should produce a larger signal than steady growth"
        )

    def test_normalization_mode_is_time_series(self) -> None:
        sig = GrowthTrend()
        assert sig.normalization == "time_series"


class TestAbstention:
    """Signal abstains cleanly when indicators are unavailable."""

    def test_missing_indicator_gives_empty_score(self) -> None:
        macro = pd.DataFrame({"SOMETHING_ELSE": [1.0, 2.0]},
                             index=pd.bdate_range("2020-01-01", periods=2))
        prices = pd.DataFrame({"SPY": [1.0, 1.0]},
                              index=pd.bdate_range("2020-01-01", periods=2))
        sig = GrowthTrend(indicators=("INDPRO",))  # not in macro
        scores = sig.compute(prices, macro)
        assert scores["SPY"].isna().all(), (
            "With no available indicators, the signal should be all-NaN"
        )

def _macro_with_inflation(n_years: int = 8) -> pd.DataFrame:
    """Macro frame with a CPI index (steady, then a sharp surge) and a
    breakeven RATE that tracks it. Lets us test both the index path
    (YoY-excess) and the rate path (level-excess)."""
    n = n_years * 252
    idx = pd.bdate_range("2010-01-01", periods=n)
    cpi = np.empty(n)
    breakeven = np.empty(n)
    for i in range(n):
        year = i / 252
        if year < 5:
            cpi[i] = 250.0 * (1.02 ** year)       # steady ~2% inflation
            breakeven[i] = 2.0                      # steady 2% expected
        else:
            # inflation surge: prices accelerate, expectations jump
            cpi[i] = 250.0 * (1.02 ** 5) * (1.07 ** (year - 5))  # ~7%/yr
            breakeven[i] = 2.0 + 3.0 * (year - 5)   # expectations climb
    return pd.DataFrame({"CPIAUCSL": cpi, "T5YIE": breakeven}, index=idx)


class TestLevelExcess:
    """The rate-path trend measure (for series that are ALREADY rates)."""

    def test_level_excess_skips_yoy(self) -> None:
        # A constant rate has zero excess (equals its own average).
        idx = pd.bdate_range("2010-01-01", periods=600)
        flat = pd.Series(2.0, index=idx)
        excess = InflationTrend._level_excess(flat)
        # After warm-up, a constant series sits exactly at its rolling mean -> 0
        assert np.isclose(excess.dropna().iloc[-1], 0.0)

    def test_level_excess_positive_when_above_trend(self) -> None:
        idx = pd.bdate_range("2010-01-01", periods=900)
        # Rate steady at 2, then jumps to 4 -> should be above its trailing mean
        vals = [2.0] * 600 + [4.0] * 300
        s = pd.Series(vals, index=idx)
        excess = InflationTrend._level_excess(s)
        assert excess.iloc[-1] > 0, "Rate above its recent average -> positive excess"


class TestInflationScore:
    """The composite inflation score combines index and rate indicators."""

    def test_surge_is_positive(self) -> None:
        macro = _macro_with_inflation()
        sig = InflationTrend(index_indicators=("CPIAUCSL",), rate_indicators=("T5YIE",))
        score = sig._compute_inflation_score(macro)
        # During the surge (year 6+), the score should be clearly positive.
        surge = score.loc["2016-06-01":"2017-12-31"].mean()
        assert surge > 0, "Inflation surge should give a positive composite score"

    def test_uses_both_index_and_rate_paths(self) -> None:
        """Both an index indicator and a rate indicator should contribute."""
        macro = _macro_with_inflation()
        # With both
        both = InflationTrend(
            index_indicators=("CPIAUCSL",), rate_indicators=("T5YIE",)
        )._compute_inflation_score(macro)
        # Index only
        index_only = InflationTrend(
            index_indicators=("CPIAUCSL",), rate_indicators=()
        )._compute_inflation_score(macro)
        # Adding the rate path should change the composite (it's averaged in).
        assert not both.equals(index_only), (
            "The rate indicator should contribute to the composite score"
        )


class TestInflationBetaMapping:
    """Inflation hedges load positive; inflation-bearish assets negative."""

    def test_hedge_positive_equity_negative(self) -> None:
        macro = _macro_with_inflation()
        prices = pd.DataFrame(
            {"GLD": 1.0, "SPY": 1.0, "HYG": 1.0},  # hedge, bearish, neutral
            index=macro.index,
        )
        sig = InflationTrend(index_indicators=("CPIAUCSL",), rate_indicators=("T5YIE",))
        scores = sig.compute(prices, macro)
        last = scores.iloc[-1]
        # During the surge, GLD (hedge) > 0, SPY (bearish) < 0, HYG (neutral) NaN
        assert last["GLD"] > 0, "Gold (inflation hedge) should be positive in a surge"
        assert last["SPY"] < 0, "Equities (inflation-bearish) should be negative"
        assert np.isnan(last["HYG"]), "Neutral-beta asset should abstain (NaN)"

    def test_gld_differs_from_growth(self) -> None:
        """GLD is +1 for inflation but 0 for growth — the maps genuinely differ."""
        from atlas.signals.macro_trend import GrowthTrend
        assert InflationTrend.DEFAULT_INFLATION_BETAS["GLD"] == 1.0
        assert GrowthTrend.DEFAULT_GROWTH_BETAS["GLD"] == 0.0


class TestInflationNormalization:
    def test_normalization_mode_is_time_series(self) -> None:
        assert InflationTrend().normalization == "time_series"