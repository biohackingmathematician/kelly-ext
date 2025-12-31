"""
Optimization module for Kelly portfolio construction.

Provides:
- Single-asset closed-form solutions
- Multi-asset SDP formulation
- Constraint handling (leverage, long-only, sector)
"""

from .single_asset import (
    kelly_single_asset,
    drk_single_asset,
    optimal_kelly_fraction,
)
from .multi_asset import (
    kelly_multi_asset,
    drk_multi_asset_sdp,
    mean_variance_kelly,
)
from .constraints import (
    LeverageConstraint,
    LongOnlyConstraint,
    SectorConstraint,
    apply_constraints,
)

__all__ = [
    # Single asset
    'kelly_single_asset',
    'drk_single_asset',
    'optimal_kelly_fraction',
    # Multi asset
    'kelly_multi_asset',
    'drk_multi_asset_sdp',
    'mean_variance_kelly',
    # Constraints
    'LeverageConstraint',
    'LongOnlyConstraint',
    'SectorConstraint',
    'apply_constraints',
]

