"""Abstract base class defining the contract every signal must fulfill.

Every concrete signal subclasses `Signal` and implements `_compute_raw`. The
base class provides shared post-processing (z-scoring, winsorization) so all
signals produce comparable, well-behaved outputs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Signal(ABC):
    VALID_NORMALIZATIONS = ("cross_sectional", "time_series")

    def __init__(
        self,
        name: str | None = None,
        normalization: str = "cross_sectional",
    ) -> None:
        # Default the signal's name to its class name if not provided.
        self.name = name or self.__class__.__name__
        if normalization not in self.VALID_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {self.VALID_NORMALIZATIONS}, "
                f"got {normalization!r}"
            )
        self.normalization = normalization

    @abstractmethod
    def _compute_raw(
        self, prices: pd.DataFrame, macro: pd.DataFrame
    ) -> pd.DataFrame:
        """Produce the raw, un-standardized signal."""
        ...

    def compute(
        self, prices: pd.DataFrame, macro: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute the final, post-processed signal, depending on 
        the normalization mode. Time-series for a single directional 
        view applied via per-asset betas (growth trend, inflation trend), 
        and cross-sectional for signals that rank assets against each other 
        on each date."""
        raw = self._compute_raw(prices, macro)

        if self.normalization == "time_series":
            # Already time-series-normalized inside _compute_raw; pass through.
            # Cross-sectional winsorize/z-score would destroy the magnitude.
            return raw

        # Default: cross-sectional path.
        winsorized = self._winsorize(raw, limit=4.0)
        standardized = self._cross_sectional_zscore(winsorized)
        return standardized

    @staticmethod
    def _winsorize(df: pd.DataFrame, limit: float = 4.0) -> pd.DataFrame:
        """Clip extreme values to ±`limit` robust standard deviations per row.

        Uses median and MAD rather than mean/std, because mean and std are themselves 
        corrupted by outliers — a single huge value inflates the std so much the 
        clipping band stretches past the outlier and fails to catch it.

        MAD is scaled by 1.4826 so that for normally-distributed data it equals
        the standard deviation, keeping `limit` interpretable as "standard
        deviations".

        Includes safeguards against situatino where more than half of the values
        are identical.
        """
        row_median = df.median(axis=1)
        abs_dev = df.sub(row_median, axis=0).abs()
        mad = abs_dev.median(axis=1)
        robust_std = mad * 1.4826

        lower = row_median - limit * robust_std
        upper = row_median + limit * robust_std

        clipped = df.clip(lower=lower, upper=upper, axis=0)

        # Where robust_std is zero (or NaN), winsorization would collapse the
        # row to its median. In those rows, return the ORIGINAL values unclipped.
        zero_spread = (robust_std == 0.0) | robust_std.isna()
        clipped = clipped.where(~zero_spread, df, axis=0)

        return clipped

    @staticmethod
    def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
        row_mean = df.mean(axis=1)
        row_std = df.std(axis=1)
        # Subtract mean and divide by std, row-wise (axis=0 broadcasts down rows)
        centered = df.sub(row_mean, axis=0)
        z = centered.div(row_std, axis=0)
        return z

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"