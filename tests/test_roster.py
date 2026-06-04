"""Tests for the blend roster (atlas.portfolio.roster).

build_roster only composes already-tested signals, so its real integration
check is the byte-identical re-run of build_composite.py on REAL data. The
keys + float64 contract is verified there. This file pins the one data-free
invariant: the scale config (a strategy constant), kept consistent with the
"known-answer inputs, no data-cache dependency" test convention.
"""
from __future__ import annotations

from atlas.portfolio.roster import ROSTER_SCALES


def test_roster_scales_targets_inflation_only():
    # InflationTrend is the only signal rescaled (time-series norm reaches ~+-4.5);
    # the cross-sectional signals (~+-2) are left at 1.0. A-priori 2.0, not fitted.
    assert ROSTER_SCALES == {"InflationTrend": 2.0}