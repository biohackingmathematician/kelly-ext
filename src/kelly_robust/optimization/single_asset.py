"""
Single-Asset Kelly Optimization.

Closed-form solutions for single-asset Kelly and DRK.

Author: Agna Chan
Date: December 2025
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class KellyResult:
    """Container for Kelly optimization result."""
    fraction: float
    expected_growth: float
    mu: float
    sigma2: float
    risk_free: float


def kelly_single_asset(
    mu: float,
    sigma2: float,
    risk_free: float = 0.0,
) -> float:
    """
    Classical Kelly fraction for a single asset.
    
    f* = (μ - r) / σ²
    
    Parameters
    ----------
    mu : float
        Expected return
    sigma2 : float
        Return variance
    risk_free : float
        Risk-free rate
        
    Returns
    -------
    float
        Optimal Kelly fraction
    """
    if sigma2 <= 0:
        raise ValueError("Variance must be positive")
    
    return (mu - risk_free) / sigma2


def drk_single_asset(
    mu_hat: float,
    sigma2_hat: float,
    epsilon: float,
    risk_free: float = 0.0,
    min_fraction: float = 0.0,
    max_fraction: float = 1.0,
) -> float:
    """
    Distributionally Robust Kelly fraction.
    
    f_DRK = (μ̂ - r - ε) / σ̂²
    
    Under Wasserstein ambiguity, the worst-case distribution
    shifts the mean adversarially by ε.
    
    Parameters
    ----------
    mu_hat : float
        Estimated mean
    sigma2_hat : float
        Estimated variance
    epsilon : float
        Ambiguity radius
    risk_free : float
        Risk-free rate
    min_fraction, max_fraction : float
        Allocation constraints
        
    Returns
    -------
    float
        Robust Kelly fraction
    """
    if sigma2_hat <= 0:
        raise ValueError("Variance must be positive")
    
    # Worst-case mean
    mu_wc = mu_hat - epsilon
    
    # Kelly on worst-case
    f_drk = (mu_wc - risk_free) / sigma2_hat
    
    # Apply constraints
    return np.clip(f_drk, min_fraction, max_fraction)


def optimal_kelly_fraction(
    returns: np.ndarray,
    risk_free: float = 0.0,
    method: str = 'plugin',
    alpha: float = 0.1,
    min_fraction: float = 0.0,
    max_fraction: float = 1.0,
) -> KellyResult:
    """
    Compute optimal Kelly fraction with specified method.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    risk_free : float
        Risk-free rate
    method : str
        'plugin', 'half', 'drk', or 'ack'
    alpha : float
        Confidence level for DRK/ACK
    min_fraction, max_fraction : float
        Allocation constraints
        
    Returns
    -------
    KellyResult
        Optimal fraction and diagnostics
    """
    mu_hat = np.mean(returns)
    sigma2_hat = np.var(returns, ddof=1)
    
    if method == 'plugin':
        f = kelly_single_asset(mu_hat, sigma2_hat, risk_free)
        f = np.clip(f, min_fraction, max_fraction)
        
    elif method == 'half':
        f = 0.5 * kelly_single_asset(mu_hat, sigma2_hat, risk_free)
        f = np.clip(f, min_fraction, max_fraction)
        
    elif method == 'drk':
        # Standard error based epsilon
        n = len(returns)
        se = np.sqrt(sigma2_hat / n)
        from scipy import stats
        z = stats.norm.ppf(1 - alpha / 2)
        epsilon = z * se
        f = drk_single_asset(mu_hat, sigma2_hat, epsilon, risk_free,
                             min_fraction, max_fraction)
        
    elif method == 'ack':
        # Use conformal calibration
        from kelly_robust.calibration.conformal import calibrate_epsilon_conformal
        epsilon, mu_hat, sigma2_hat = calibrate_epsilon_conformal(returns, alpha)
        f = drk_single_asset(mu_hat, sigma2_hat, epsilon, risk_free,
                             min_fraction, max_fraction)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute expected growth rate
    g = risk_free + f * (mu_hat - risk_free) - 0.5 * f**2 * sigma2_hat
    
    return KellyResult(
        fraction=f,
        expected_growth=g,
        mu=mu_hat,
        sigma2=sigma2_hat,
        risk_free=risk_free,
    )

