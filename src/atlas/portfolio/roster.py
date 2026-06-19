"""Single source of truth for the 4-signal blend roster.

GrowthTrend is intentionally absent: dropped from the blend in Phase 2 (COVID
publication-lag whipsaw); reintroduce only after a Phase 3 nowcasting layer.
"""
from __future__ import annotations

import pandas as pd

from atlas.signals.momentum import TSMomentum
from atlas.signals.carry import BondCarry, FXCarry
from atlas.signals.macro_trend import InflationTrend

# InflationTrend is time-series normalized (~+-4.5); the other three are
# cross-sectionally z-scored (~+-2). The divisor is a shape-preserving change of
# units putting it on comparable footing for blending -- an a-priori "2 sigma is
# strong" choice, NOT fitted to the observed peak. Passed to blend_signals.
ROSTER_SCALES = {"InflationTrend": 2.0}


def build_roster(
    prices: pd.DataFrame,
    prices_unadjusted: pd.DataFrame,
    dividends: pd.DataFrame,
    macro: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "TSMomentum": TSMomentum().compute(prices, macro),
        "BondCarry": BondCarry(
            dividends=dividends, prices_unadjusted=prices_unadjusted
        ).compute(prices, macro),
        "FXCarry": FXCarry().compute(prices, macro),
        "InflationTrend": InflationTrend().compute(prices, macro),
    }