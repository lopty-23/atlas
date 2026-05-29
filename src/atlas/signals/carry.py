"""Carry signals: bond carry (yield - short rate) and FX carry (rate differential)."""
from __future__ import annotations

import pandas as pd

from atlas.data.returns import distribution_yield
from atlas.signals.base import Signal

# Short-rate proxy used as the funding leg for both bond and FX carry.
US_SHORT_RATE = "DGS3MO"


class BondCarry(Signal):
    """Bond carry: ETF yield minus the short rate. Distribution-yield path for 
    IEF, TLT, LQD, HYG, EMB; curve-carry path for BWX.

    Each dist-yield ETF abstains (NaN) until a FULL trailing window has
    elapsed since its first ex-date. 
    """

    DEFAULT_MATURITY_MAP = {
        "IEF": "DGS10",
        "TLT": "DGS30",
        "LQD": "DGS10",
        "HYG": "DGS10",
        "BWX": "DGS10",
        "EMB": "DGS10",
    }

    DEFAULT_DIST_YIELD_ETFS = ("IEF", "TLT", "LQD", "HYG", "EMB")

    def __init__(
        self,
        dividends: pd.DataFrame | None = None,
        prices_unadjusted: pd.DataFrame | None = None,
        maturity_map: dict[str, str] | None = None,
        dist_yield_etfs: tuple[str, ...] | None = None,
        short_rate: str = US_SHORT_RATE,
        yield_window_days: int = 365,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.dividends = dividends
        self.prices_unadjusted = prices_unadjusted
        self.maturity_map = maturity_map or dict(self.DEFAULT_MATURITY_MAP)
        self.dist_yield_etfs = (
            tuple(dist_yield_etfs)
            if dist_yield_etfs is not None
            else self.DEFAULT_DIST_YIELD_ETFS
        )
        self.short_rate = short_rate
        self.yield_window_days = yield_window_days

        if self.dist_yield_etfs and (dividends is None or prices_unadjusted is None):
            raise ValueError(
                "dist_yield_etfs is non-empty but dividends and/or "
                "prices_unadjusted were not provided"
            )

        # Test for missing dividend or unadjusted price columns for the configured dist-yield ETFs.
        if self.dist_yield_etfs:
            assert dividends is not None and prices_unadjusted is not None
            missing_divs = [t for t in self.dist_yield_etfs if t not in dividends.columns]
            missing_px = [
                t for t in self.dist_yield_etfs
                if t not in prices_unadjusted.columns
            ]
            if missing_divs or missing_px:
                raise ValueError(
                    "dist_yield_etfs missing required data -- "
                    f"no dividend column for {missing_divs!r}, "
                    f"no unadjusted-price column for {missing_px!r}"
                )

    def _compute_raw(self, prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        if self.short_rate not in macro.columns:
            return out

        short_decimal = (
            macro[self.short_rate].reindex(prices.index, method="ffill") / 100.0
        )

        # Distribution-yield path
        dist_yields: pd.DataFrame | None = None
        if self.dist_yield_etfs:
            assert self.dividends is not None and self.prices_unadjusted is not None
            dist_yields = distribution_yield(
                self.dividends,
                self.prices_unadjusted,
                window_days=self.yield_window_days,
            )
            # Abstain (NaN) until a full trailing window has elapsed since
            # the first ex-date. 
            warmup = pd.Timedelta(days=self.yield_window_days)
            for etf in self.dist_yield_etfs:
                if etf not in dist_yields.columns:
                    continue  # not enough overlap to produce a yield series
                first_div = self.dividends[etf].dropna().index.min()
                if pd.notna(first_div):
                    warmup_end = first_div + warmup
                    dist_yields.loc[dist_yields.index < warmup_end, etf] = pd.NA

        for etf, yield_series in self.maturity_map.items():
            if etf not in prices.columns:
                continue

            if etf in self.dist_yield_etfs:
                # Construction-time validation guarantees dist_yields is set
                # and contains a column for every dist-yield ETF.
                assert dist_yields is not None
                yield_decimal = dist_yields[etf].reindex(prices.index, method="ffill")
                out[etf] = yield_decimal - short_decimal
            else:
                if yield_series not in macro.columns:
                    continue
                long_yield_decimal = (
                    macro[yield_series].reindex(prices.index, method="ffill") / 100.0
                )
                out[etf] = long_yield_decimal - short_decimal

        return out


class FXCarry(Signal):
    """FX carry: foreign short rate minus USD short rate, per currency ETF.

    UUP (long USD index) is treated as the inverse: USD rate minus the average
    of the available foreign rates.
    """

    DEFAULT_RATE_MAP = {
        "FXE": "IR3TIB01EZM156N",  # EUR
        "FXY": "IR3TIB01JPM156N",  # JPY
        "FXA": "IR3TIB01AUM156N",  # AUD
    }

    def __init__(
        self,
        rate_map: dict[str, str] | None = None,
        usd_rate: str = US_SHORT_RATE,
        usd_etf: str = "UUP",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.rate_map = rate_map or self.DEFAULT_RATE_MAP
        self.usd_rate = usd_rate
        self.usd_etf = usd_etf

    def _compute_raw(self, prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        if self.usd_rate not in macro.columns:
            return out

        usd = macro[self.usd_rate].reindex(prices.index, method="ffill")

        foreign_rates_used: list[str] = []
        for etf, rate_series in self.rate_map.items():
            if etf not in prices.columns or rate_series not in macro.columns:
                continue
            foreign = macro[rate_series].reindex(prices.index, method="ffill")
            out[etf] = foreign - usd
            foreign_rates_used.append(rate_series)

        if self.usd_etf in prices.columns and foreign_rates_used:
            avg_foreign = (
                macro[foreign_rates_used]
                .reindex(prices.index, method="ffill")
                .mean(axis=1)
            )
            out[self.usd_etf] = usd - avg_foreign

        return out
