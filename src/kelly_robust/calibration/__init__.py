"""
Calibration methods for ambiguity radius.

This module provides multiple approaches to calibrate the ambiguity
radius ε for Distributionally Robust Kelly optimization:

1. Conformal Prediction (distribution-free, finite-sample guarantees)
2. Bootstrap (resampling-based uncertainty quantification)
3. Bayesian (posterior-based credible intervals)
"""

from .conformal import (
    calibrate_epsilon_conformal,
    split_conformal_calibration,
    cross_conformal_calibration,
    adaptive_conformal_interval,
)
from .bootstrap import (
    calibrate_epsilon_bootstrap,
    parametric_bootstrap,
    block_bootstrap,
    stationary_bootstrap,
)
from .bayesian import (
    calibrate_epsilon_bayesian,
    normal_gamma_posterior,
    compute_posterior_kelly,
)

__all__ = [
    # Conformal
    'calibrate_epsilon_conformal',
    'split_conformal_calibration',
    'cross_conformal_calibration',
    'adaptive_conformal_interval',
    # Bootstrap
    'calibrate_epsilon_bootstrap',
    'parametric_bootstrap',
    'block_bootstrap',
    'stationary_bootstrap',
    # Bayesian
    'calibrate_epsilon_bayesian',
    'normal_gamma_posterior',
    'compute_posterior_kelly',
]

