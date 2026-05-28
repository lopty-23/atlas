"""Time-series (trend) momentum signals.

Time-series momentum: assets that have risen over a trailing window tend to
keep rising. The raw signal is the trailing return, optionally scaled by the
asset's realized volatility so that trends in low-vol assets (bonds) are
comparable to trends in high-vol assets (equities, commodities).

Reference: Moskowitz, Ooi, Pedersen (2012), "Time Series Momentum", JFE.
"""
from __future__ import annotations

import pandas as pd

from atlas.data.returns import realized_volatility, simple_returns, trailing_returns
from atlas.signals.base import Signal


class TSMomentum(Signal):
    def __init__(
        self,
        lookback_days: int = 252,
        skip_days: int = 0,
        vol_scale: bool = True,
        vol_window: int = 63,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.vol_scale = vol_scale
        self.vol_window = vol_window

    def _compute_raw(self, prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
        if self.skip_days > 0:
            past = prices.shift(self.skip_days)
            mom = past.pct_change(periods=self.lookback_days - self.skip_days)
        else:
            mom = trailing_returns(prices, lookback=self.lookback_days)

        if self.vol_scale:
            daily_rets = simple_returns(prices)
            vol = realized_volatility(daily_rets, window=self.vol_window, annualize=True)
            # Avoid division by zero / tiny vol blowing up the signal.
            mom = mom / vol.replace(0.0, pd.NA)

        return mom