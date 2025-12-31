"""
Simulation module for return generation and Monte Carlo analysis.

Provides various return models:
- Geometric Brownian Motion (GBM)
- Student-t distributed returns
- GARCH volatility models
- Jump-diffusion (Merton model)
- Regime-switching models

Also provides Monte Carlo utilities for strategy evaluation.
"""

from .return_models import (
    simulate_gbm_returns,
    simulate_t_returns,
    simulate_garch_returns,
    simulate_jump_diffusion,
    simulate_regime_switching,
    create_return_simulator,
)
from .monte_carlo import (
    simulate_wealth_paths,
    compute_growth_metrics,
    monte_carlo_kelly_comparison,
    parameter_sensitivity_analysis,
)

__all__ = [
    # Return models
    'simulate_gbm_returns',
    'simulate_t_returns',
    'simulate_garch_returns',
    'simulate_jump_diffusion',
    'simulate_regime_switching',
    'create_return_simulator',
    # Monte Carlo
    'simulate_wealth_paths',
    'compute_growth_metrics',
    'monte_carlo_kelly_comparison',
    'parameter_sensitivity_analysis',
]

