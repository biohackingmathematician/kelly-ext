"""
Distributionally Robust Kelly (DRK) Optimization

A mathematically rigorous framework for optimal growth-rate portfolio 
allocation under distributional ambiguity.

Key Features:
- Classical Kelly fraction computation (single and multi-asset)
- Distributionally Robust Kelly under Wasserstein ambiguity
- Conformal prediction for ambiguity calibration
- Walk-forward backtesting with transaction costs
- Monte Carlo simulation with various return models

Main Entry Points:
    from kelly_robust import adaptive_conformal_kelly, run_kelly_comparison
    from kelly_robust.backtest import WalkForwardBacktest
    from kelly_robust.data import download_prices

Author: Agna Chan
Affiliation: Columbia University, Department of Statistics
Date: December 2025
"""

__version__ = "0.1.0"
__author__ = "Agna Chan"

# Core Kelly functions
from .core.kelly import (
    kelly_single_asset,
    kelly_multi_asset,
    growth_rate_gaussian,
    growth_rate_exact,
    drk_single_asset_closed_form,
    calibrate_epsilon_conformal,
    adaptive_conformal_kelly,
    drk_multi_asset_socp,
    drk_multi_asset_sdp,  # Deprecated alias
    simulate_gbm_returns,
    simulate_wealth_paths,
    compute_growth_metrics,
    run_kelly_comparison,
    DRKResult,
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    # Core functions
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

