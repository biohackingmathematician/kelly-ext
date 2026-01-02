"""
Core Kelly optimization functions.

This module contains the fundamental Kelly criterion implementations:
- Classical Kelly (single and multi-asset)
- Growth rate computation
- Distributionally Robust Kelly (DRK)
- Conformal calibration
"""

from .kelly import (
    kelly_single_asset,
    kelly_multi_asset,
    growth_rate_gaussian,
    growth_rate_exact,
    drk_single_asset_closed_form,
    calibrate_epsilon_conformal,
    adaptive_conformal_kelly,
    drk_multi_asset_socp,
    drk_multi_asset_sdp,  # Deprecated alias for backward compatibility
    simulate_gbm_returns,
    simulate_wealth_paths,
    compute_growth_metrics,
    run_kelly_comparison,
    DRKResult,
)

__all__ = [
    'kelly_single_asset',
    'kelly_multi_asset',
    'growth_rate_gaussian',
    'growth_rate_exact',
    'drk_single_asset_closed_form',
    'calibrate_epsilon_conformal',
    'adaptive_conformal_kelly',
    'drk_multi_asset_socp',
    'drk_multi_asset_sdp',  # Deprecated alias
    'simulate_gbm_returns',
    'simulate_wealth_paths',
    'compute_growth_metrics',
    'run_kelly_comparison',
    'DRKResult',
]

