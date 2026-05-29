"""Macro-trend signals: growth trend and (later) inflation trend.

These signals compute one score about the economy as a time series, from 
point-in-time macro indicators, then map that single score onto assets via 
fixed betas encoding how each asset class responds to the macro variable.
  
The beta signs follow AQR's "Economic Trend" methodology (Brooks &
Feilbogen, 2023): increasing growth is bullish for equities (cash-flow),
commodities (demand), and a country's own currency (Balassa-Samuelson), and
bearish for government bonds (inflation + real-rate pressure). Because our
growth indicators are US-specific, we map "rising US growth" to long USD
(UUP +) and short foreign currencies (FXE/FXY/FXA -).

Trend measure: "excess growth" — year-over-year growth minus its own
trailing multi-year average (Macrosynergy/JPMaQS convention). This is
robust to the base effects that make the simpler year-over-year-of-
year-over-year "acceleration" measure explode during shocks. Each
indicator's trend is z-scored over time using an expanding window, 
then the indicators are averaged into a composite.
"""
from __future__ import annotations

import pandas as pd

from atlas.signals.base import Signal

# Trading days per year, for the YoY computation on a daily-aligned series.
TRADING_DAYS = 252


class GrowthTrend(Signal):
    DEFAULT_INDICATORS = ("INDPRO", "PAYEMS", "GDPC1")

    DEFAULT_GROWTH_BETAS = {
        # Equities: pro-cyclical (cash-flow impact)
        "SPY": 1.0, "IWM": 1.0, "EFA": 1.0, "EEM": 1.0,
        # Industrial commodities: pro-cyclical (demand)
        "DBC": 1.0, "USO": 1.0, "BNO": 1.0, "SLV": 1.0,
        # Credit: pro-cyclical (spread-driven, risk-on)
        "HYG": 1.0, "EMB": 1.0,
        # Real estate: pro-cyclical (cash-flow dominates)
        "VNQ": 1.0, "RWX": 1.0,
        # USD: rising US growth -> strong dollar
        "UUP": 1.0,
        # Government bonds: counter-cyclical (inflation + real-rate pressure)
        "IEF": -1.0, "TLT": -1.0,
        # Foreign currencies: weaken vs USD on US growth
        "FXE": -1.0, "FXY": -1.0, "FXA": -1.0,
        # Neutral (beta 0 -> abstain): GLD (monetary asset), LQD (rate vs
        # spread offset), BWX (foreign duration, US-growth view murky)
        "GLD": 0.0, "LQD": 0.0, "BWX": 0.0,
    }

    def __init__(
        self,
        indicators: tuple[str, ...] | None = None,
        growth_betas: dict[str, float] | None = None,
        min_periods: int = 252,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, normalization="time_series")
        self.indicators = indicators if indicators is not None else self.DEFAULT_INDICATORS
        self.growth_betas = growth_betas or dict(self.DEFAULT_GROWTH_BETAS)
        self.min_periods = min_periods

    @staticmethod
    def _yoy_excess(series: pd.Series, baseline_window: int = 3 * TRADING_DAYS) -> pd.Series:
        yoy = series / series.shift(TRADING_DAYS) - 1.0
        baseline = yoy.rolling(window=baseline_window, min_periods=TRADING_DAYS).mean()
        return yoy - baseline
    
    @staticmethod
    def _expanding_zscore(series: pd.Series, min_periods: int) -> pd.Series:
        """Time-series z-score using an EXPANDING window (no look-ahead).

        At each date t, standardize using the mean and std of all values up to
        and including t — never future values. This is the critical
        no-look-ahead property: a full-sample z-score would leak future info.
        """
        expanding_mean = series.expanding(min_periods=min_periods).mean()
        expanding_std = series.expanding(min_periods=min_periods).std()
        return (series - expanding_mean) / expanding_std

    def _compute_growth_score(self, macro: pd.DataFrame) -> pd.Series:
        """Composite growth score: average of each indicator's z-scored trend."""
        zscored_trends = []
        for indicator in self.indicators:
            if indicator not in macro.columns:
                continue
            trend = self._yoy_excess(macro[indicator])
            z = self._expanding_zscore(trend, self.min_periods)
            zscored_trends.append(z)

        if not zscored_trends:
            # No indicators available -> empty score (all NaN on macro index)
            return pd.Series(index=macro.index, dtype=float)

        # Average the z-scored trends into one composite (equal weight).
        composite = pd.concat(zscored_trends, axis=1).mean(axis=1)
        return composite

    def _compute_raw(self, prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

        growth_score = self._compute_growth_score(macro)
        if growth_score.isna().all():
            return out  # nothing computable -> all NaN

        # Align the macro-derived score onto the price calendar.
        score_aligned = growth_score.reindex(prices.index, method="ffill")

        # Map the single score onto assets via their betas.
        for asset, beta in self.growth_betas.items():
            if asset not in prices.columns or beta == 0.0:
                continue  # not tradeable here, or neutral -> abstain (NaN)
            out[asset] = score_aligned * beta

        return out