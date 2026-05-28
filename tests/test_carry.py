"""Tests for BondCarry and FXCarry.

Coverage:
    - BondCarry distribution-yield path: arithmetic correctness, ETF
      differentiation, point-in-time safety (future dividends and future
      prices must not affect earlier signal values), and pre-inception
      abstention (NaN before the first ex-date).
    - BondCarry curve path: BWX uses the yield curve, not dividends.
    - FXCarry: foreign-minus-USD arithmetic for long-foreign ETFs and the
      USD-minus-average mechanic for UUP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas.signals.carry import BondCarry, FXCarry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_prices(tickers: list[str], n: int = 30, level: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({t: [level] * n for t in tickers}, index=idx, dtype=float)


def _make_macro(idx: pd.DatetimeIndex, **series: float) -> pd.DataFrame:
    return pd.DataFrame({k: [v] * len(idx) for k, v in series.items()}, index=idx)


# ---------------------------------------------------------------------------
# BondCarry: arithmetic
# ---------------------------------------------------------------------------


class TestBondCarryArithmetic:
    """Carry math is verified against hand-computed values."""

    def test_dist_yield_path_known_value(self) -> None:
        """In steady state, carry = (trailing dividends / price) - short_rate.

        With monthly $1 dividends on a $100 price and short rate 3%, the
        trailing-12m window contains ~12 payments at any post-warm-up
        date, so carry = (~$12 / $100) - 0.03 = ~9%. We don't hardcode 12
        -- we derive the expected count from the fixture and assert the
        implementation matches.
        """
        prices = _make_prices(["LQD"], n=800)  # ~3.2 years of bdays
        prices_unadj = prices.copy()

        # Monthly $1 dividends spanning the FULL price range, paced at 22
        # bdays (~1 month). Picking dates from prices.index guarantees they
        # all survive the reindex onto the business-day calendar.
        div_dates = pd.DatetimeIndex(
            [prices.index[i] for i in range(0, len(prices), 22)]
        )
        dividends = pd.DataFrame({"LQD": [1.0] * len(div_dates)}, index=div_dates)
        macro = _make_macro(prices.index, DGS3MO=3.0)  # FRED yields in %

        sig = BondCarry(
            dividends=dividends,
            prices_unadjusted=prices_unadj,
            dist_yield_etfs=("LQD",),
            maturity_map={"LQD": "DGS10"},
        )
        raw = sig._compute_raw(prices, macro)

        # Reconstruct the expected carry at the last row directly from the
        # fixture using the same window semantics as the implementation
        # (rolling("365D") is right-closed, left-open).
        last_date = raw.index[-1]
        window_start = last_date - pd.Timedelta(days=365)
        n_in_window = int(
            ((dividends.index > window_start) & (dividends.index <= last_date)).sum()
        )
        expected_carry = (n_in_window * 1.0 / 100.0) - (3.0 / 100.0)

        assert np.isclose(raw["LQD"].loc[last_date], expected_carry), (
            f"with {n_in_window} divs in window, "
            f"expected carry {expected_carry:.6f}, got {raw['LQD'].loc[last_date]:.6f}"
        )
        # Fixture sanity: a 22-bday cadence (~31 calendar days/div) puts
        # ~12 dividends in a 365-day window.
        assert n_in_window in (11, 12, 13), (
            f"fixture drift -- expected ~12 monthly divs in trailing 12m, "
            f"got {n_in_window}"
        )

    def test_curve_path_known_value(self) -> None:
        """BWX with DGS10=4.5%, DGS3MO=3.0% must produce 1.5% curve carry."""
        prices = _make_prices(["BWX"], n=10)
        macro = _make_macro(prices.index, DGS10=4.5, DGS3MO=3.0)

        sig = BondCarry(
            dividends=pd.DataFrame(),  # irrelevant for BWX
            prices_unadjusted=prices.copy(),
            dist_yield_etfs=(),  # BWX uses curve carry
            maturity_map={"BWX": "DGS10"},
        )
        raw = sig._compute_raw(prices, macro)
        assert np.isclose(raw["BWX"].iloc[-1], 0.015)

    def test_short_rate_subtraction_consistent(self) -> None:
        """Doubling the short rate must lower carry by the rate increment."""
        prices = _make_prices(["LQD"], n=400)
        div_dates = pd.to_datetime(["2024-06-01"])
        dividends = pd.DataFrame({"LQD": [4.0]}, index=div_dates)  # 4% yield

        macro_low = _make_macro(prices.index, DGS3MO=2.0)
        macro_high = _make_macro(prices.index, DGS3MO=4.0)

        sig = BondCarry(
            dividends=dividends,
            prices_unadjusted=prices.copy(),
            dist_yield_etfs=("LQD",),
            maturity_map={"LQD": "DGS10"},
        )
        raw_low = sig._compute_raw(prices, macro_low)["LQD"].iloc[-1]
        raw_high = sig._compute_raw(prices, macro_high)["LQD"].iloc[-1]
        assert np.isclose(raw_high - raw_low, -0.02)


# ---------------------------------------------------------------------------
# BondCarry: differentiation
# ---------------------------------------------------------------------------


class TestBondCarryDifferentiation:
    """The whole point of the rewrite: bond ETFs no longer collapse to one value."""

    def test_etfs_with_different_yields_get_different_carries(self) -> None:
        prices = _make_prices(["IEF", "HYG", "EMB"], n=800)
        prices_unadj = prices.copy()

        # Monthly dividends at different magnitudes spanning the FULL price
        # range: IEF $0.20, HYG $0.60, EMB $0.50 (-> roughly 2.4%, 7.2%,
        # 6% annual yields at $100 price).
        div_dates = pd.DatetimeIndex(
            [prices.index[i] for i in range(0, len(prices), 22)]
        )
        n = len(div_dates)
        dividends = pd.DataFrame(
            {"IEF": [0.20] * n, "HYG": [0.60] * n, "EMB": [0.50] * n},
            index=div_dates,
        )
        macro = _make_macro(prices.index, DGS3MO=3.0, DGS10=4.0)

        sig = BondCarry(
            dividends=dividends,
            prices_unadjusted=prices_unadj,
            dist_yield_etfs=("IEF", "HYG", "EMB"),
            maturity_map={"IEF": "DGS10", "HYG": "DGS10", "EMB": "DGS10"},
        )
        raw = sig._compute_raw(prices, macro).iloc[-1]
        vals = {t: raw[t] for t in ["IEF", "HYG", "EMB"]}

        # HYG > EMB > IEF strictly.
        assert vals["HYG"] > vals["EMB"] > vals["IEF"]
        # No two are equal -- the bug the rewrite fixes.
        assert not np.isclose(vals["HYG"], vals["EMB"])
        assert not np.isclose(vals["EMB"], vals["IEF"])


# ---------------------------------------------------------------------------
# BondCarry: point-in-time safety
# ---------------------------------------------------------------------------


class TestBondCarryPointInTime:
    """Future data must never leak into earlier signal values."""

    def _baseline_inputs(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # ~3.6 years of bdays with monthly dividends from day 0 so there's
        # a substantial post-warm-up region available for t_observe.
        prices = _make_prices(["LQD"], n=900)
        prices_unadj = prices.copy()
        div_dates = pd.DatetimeIndex(
            [prices.index[i] for i in range(0, len(prices), 22)]
        )
        dividends = pd.DataFrame(
            {"LQD": [1.0 / 3.0] * len(div_dates)}, index=div_dates
        )
        macro = _make_macro(prices.index, DGS3MO=3.0)
        return prices, prices_unadj, dividends, macro

    def test_future_dividend_does_not_change_earlier_signal(self) -> None:
        prices, prices_unadj, divs_clean, macro = self._baseline_inputs()
        # Pick a t_observe well past first_div + 365d so the warm-up mask
        # is over and a real (non-NaN) signal value is being observed.
        t_observe = prices.index[420]

        sig_clean = BondCarry(
            dividends=divs_clean,
            prices_unadjusted=prices_unadj,
            dist_yield_etfs=("LQD",),
            maturity_map={"LQD": "DGS10"},
        )
        raw_clean = sig_clean._compute_raw(prices, macro)
        carry_at_t = raw_clean["LQD"].loc[t_observe]

        # Inject a huge dividend AFTER t_observe.
        future_date = prices.index[-30]
        assert future_date > t_observe
        divs_corrupt = pd.concat([
            divs_clean,
            pd.DataFrame({"LQD": [50.0]}, index=[future_date]),
        ]).sort_index()
        sig_corrupt = BondCarry(
            dividends=divs_corrupt,
            prices_unadjusted=prices_unadj,
            dist_yield_etfs=("LQD",),
            maturity_map={"LQD": "DGS10"},
        )
        raw_corrupt = sig_corrupt._compute_raw(prices, macro)

        assert np.isclose(raw_corrupt["LQD"].loc[t_observe], carry_at_t), (
            "Future dividend leaked into earlier signal"
        )

    def test_future_price_does_not_change_earlier_signal(self) -> None:
        prices, prices_unadj, dividends, macro = self._baseline_inputs()
        # Past the warm-up window so we're observing a real signal value.
        t_observe = prices.index[420]

        sig_clean = BondCarry(
            dividends=dividends,
            prices_unadjusted=prices_unadj,
            dist_yield_etfs=("LQD",),
            maturity_map={"LQD": "DGS10"},
        )
        carry_at_t = sig_clean._compute_raw(prices, macro)["LQD"].loc[t_observe]

        # Slam the price up by 10x AFTER t_observe.
        prices_unadj_corrupt = prices_unadj.copy()
        future_mask = prices_unadj_corrupt.index > t_observe
        prices_unadj_corrupt.loc[future_mask, "LQD"] *= 10.0

        sig_corrupt = BondCarry(
            dividends=dividends,
            prices_unadjusted=prices_unadj_corrupt,
            dist_yield_etfs=("LQD",),
            maturity_map={"LQD": "DGS10"},
        )
        carry_at_t_corrupt = sig_corrupt._compute_raw(prices, macro)["LQD"].loc[t_observe]
        assert np.isclose(carry_at_t_corrupt, carry_at_t), (
            "Future price leaked into earlier signal"
        )


# ---------------------------------------------------------------------------
# BondCarry: pre-inception abstention
# ---------------------------------------------------------------------------


class TestBondCarryAbstention:
    """Dist-yield ETFs abstain (NaN) until a full window has elapsed.

    Three regions to verify:
      (a) before the first ex-date           -> NaN (no data yet)
      (b) during the warm-up window           -> NaN (partial trailing sum
                                                  understates true yield)
      (c) after warm-up + a small buffer      -> non-NaN (full year of
                                                  dividends accumulated)
    """

    def test_warm_up_masked_through_full_window(self) -> None:
        # ~3.2 years of business days so there's room for a pre-div region,
        # a 365-day warm-up, and a post-warm-up region.
        prices = _make_prices(["HYG"], n=800)
        prices_unadj = prices.copy()

        # First dividend ~5 months in, then a monthly cadence (~22 bdays
        # apart) for 30 payments. Picking dates from prices.index guarantees
        # they all survive the reindex onto a business-day calendar.
        first_idx = 110
        div_dates = pd.DatetimeIndex(
            [prices.index[first_idx + 22 * k] for k in range(30)]
        )
        dividends = pd.DataFrame(
            {"HYG": [4.0 / 12.0] * len(div_dates)}, index=div_dates
        )
        macro = _make_macro(prices.index, DGS3MO=3.0)

        sig = BondCarry(
            dividends=dividends,
            prices_unadjusted=prices_unadj,
            dist_yield_etfs=("HYG",),
            maturity_map={"HYG": "DGS10"},
        )
        raw = sig._compute_raw(prices, macro)

        first_div = dividends.index.min()
        warmup_end = first_div + pd.Timedelta(days=sig.yield_window_days)

        before = raw["HYG"].loc[raw.index < first_div]
        in_warmup = raw["HYG"].loc[
            (raw.index >= first_div) & (raw.index < warmup_end)
        ]
        # Buffer past warmup_end so the very first dividend hasn't just
        # fallen out of the rolling window (rolling("365D") is right-closed,
        # so the value at the left edge of the window is excluded).
        post_warmup = raw["HYG"].loc[
            raw.index >= warmup_end + pd.Timedelta(days=30)
        ]

        assert before.isna().all(), "Rows before the first ex-date must be NaN"
        assert in_warmup.isna().all(), (
            "Rows during the warm-up window must be NaN -- partial trailing "
            "sums would otherwise emit an understated annual yield"
        )
        assert post_warmup.notna().any(), (
            "Rows past the warm-up window must produce real (non-NaN) carry"
        )


# ---------------------------------------------------------------------------
# BondCarry: BWX uses curve carry
# ---------------------------------------------------------------------------


class TestBondCarryBWX:
    """BWX must compute from the yield curve, not from dividends."""

    def test_bwx_independent_of_dividends(self) -> None:
        prices = _make_prices(["BWX"], n=20)
        prices_unadj = prices.copy()
        macro = _make_macro(prices.index, DGS10=5.0, DGS3MO=2.0)

        # Two BondCarry instances with wildly different (or absent) BWX
        # dividend histories. BWX is NOT in dist_yield_etfs, so curve carry
        # is the only path -- the dividends argument should be ignored
        # for BWX entirely.
        sig_no_divs = BondCarry(
            dividends=pd.DataFrame(),
            prices_unadjusted=prices_unadj,
            dist_yield_etfs=(),
            maturity_map={"BWX": "DGS10"},
        )
        sig_with_divs = BondCarry(
            dividends=pd.DataFrame(
                {"BWX": [99.0]}, index=pd.to_datetime(["2024-01-15"])
            ),
            prices_unadjusted=prices_unadj,
            dist_yield_etfs=(),
            maturity_map={"BWX": "DGS10"},
        )
        a = sig_no_divs._compute_raw(prices, macro)["BWX"].iloc[-1]
        b = sig_with_divs._compute_raw(prices, macro)["BWX"].iloc[-1]
        assert np.isclose(a, b)
        # And the value is the curve spread (5% - 2%) / 100 = 0.03.
        assert np.isclose(a, 0.03)


# ---------------------------------------------------------------------------
# BondCarry: input-validation
# ---------------------------------------------------------------------------


class TestBondCarryConstructor:
    def test_raises_when_dist_yield_etfs_but_no_data(self) -> None:
        with pytest.raises(ValueError, match="dist_yield_etfs"):
            BondCarry(dist_yield_etfs=("LQD",))

    def test_curve_only_works_without_dividend_data(self) -> None:
        # No dividends/prices_unadjusted provided, but dist_yield_etfs is
        # empty -- this is a pure curve-carry signal and should construct.
        sig = BondCarry(dist_yield_etfs=(), maturity_map={"IEF": "DGS10"})
        prices = _make_prices(["IEF"], n=5)
        macro = _make_macro(prices.index, DGS10=4.0, DGS3MO=2.0)
        raw = sig._compute_raw(prices, macro)
        assert np.isclose(raw["IEF"].iloc[-1], 0.02)


# ---------------------------------------------------------------------------
# FXCarry
# ---------------------------------------------------------------------------


class TestFXCarry:
    def test_foreign_minus_usd(self) -> None:
        """FXE carry = EUR 3m rate - USD 3m rate."""
        prices = _make_prices(["FXE"], n=10)
        macro = _make_macro(prices.index, IR3TIB01EZM156N=3.5, DGS3MO=2.0)

        sig = FXCarry(rate_map={"FXE": "IR3TIB01EZM156N"}, usd_etf="UUP")
        raw = sig._compute_raw(prices, macro)
        assert np.isclose(raw["FXE"].iloc[-1], 1.5)

    def test_uup_is_usd_minus_avg_foreign(self) -> None:
        """UUP carry = USD rate - average of available foreign rates."""
        prices = _make_prices(["FXE", "FXY", "UUP"], n=10)
        macro = _make_macro(
            prices.index,
            IR3TIB01EZM156N=4.0,
            IR3TIB01JPM156N=0.0,
            DGS3MO=2.5,
        )
        sig = FXCarry(
            rate_map={
                "FXE": "IR3TIB01EZM156N",
                "FXY": "IR3TIB01JPM156N",
            },
            usd_etf="UUP",
        )
        raw = sig._compute_raw(prices, macro)
        # avg foreign = (4 + 0) / 2 = 2, USD - avg = 2.5 - 2 = 0.5
        assert np.isclose(raw["UUP"].iloc[-1], 0.5)
