"""
Multi-Asset Kelly Optimization.

SDP formulation for multi-asset DRK and related methods.

Author: Agna Chan
Date: December 2025
"""

import numpy as np
from typing import Optional, List
import warnings


def kelly_multi_asset(
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_free: float = 0.0,
) -> np.ndarray:
    """
    Classical Kelly portfolio for multiple assets.
    
    F* = Σ⁻¹(μ - r·1)
    
    Parameters
    ----------
    mu : np.ndarray
        Expected returns (d,)
    Sigma : np.ndarray
        Covariance matrix (d, d)
    risk_free : float
        Risk-free rate
        
    Returns
    -------
    np.ndarray
        Optimal Kelly weights (d,)
    """
    d = len(mu)
    excess = mu - risk_free * np.ones(d)
    
    return np.linalg.solve(Sigma, excess)


def drk_multi_asset_sdp(
    mu_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    epsilon: float,
    risk_free: float = 0.0,
    leverage_limit: float = 1.0,
    long_only: bool = True,
    max_position: Optional[float] = None,
) -> np.ndarray:
    """
    Multi-asset DRK via semidefinite programming.
    
    Solves:
        max_f  f'(μ̂-r) - (1/2)f'Σ̂f - ε||f||_2
    
    This is the worst-case growth rate over the Wasserstein
    ambiguity set.
    
    Parameters
    ----------
    mu_hat : np.ndarray
        Estimated mean returns (d,)
    Sigma_hat : np.ndarray
        Estimated covariance (d, d)
    epsilon : float
        Ambiguity radius
    risk_free : float
        Risk-free rate
    leverage_limit : float
        Maximum sum of weights
    long_only : bool
        Enforce non-negative weights
    max_position : float, optional
        Maximum weight per asset
        
    Returns
    -------
    np.ndarray
        Robust portfolio weights (d,)
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError(
            "cvxpy required for multi-asset DRK. "
            "Install via: pip install cvxpy"
        )
    
    d = len(mu_hat)
    
    # Decision variable
    f = cp.Variable(d)
    
    # Excess returns
    excess = mu_hat - risk_free * np.ones(d)
    
    # Worst-case growth rate
    # g_wc = f'(μ-r) - (1/2)f'Σf - ε||f||_2
    growth_wc = f @ excess - 0.5 * cp.quad_form(f, Sigma_hat) - epsilon * cp.norm(f, 2)
    
    # Constraints
    constraints = [cp.sum(f) <= leverage_limit]
    
    if long_only:
        constraints.append(f >= 0)
    
    if max_position is not None:
        constraints.append(f <= max_position)
    
    # Solve
    problem = cp.Problem(cp.Maximize(growth_wc), constraints)
    
    try:
        # Try MOSEK first (best for SDP)
        problem.solve(solver=cp.MOSEK, verbose=False)
    except (cp.error.SolverError, Exception):
        try:
            # Fall back to SCS
            problem.solve(solver=cp.SCS, verbose=False)
        except (cp.error.SolverError, Exception):
            # Last resort: ECOS
            problem.solve(solver=cp.ECOS, verbose=False)
    
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        warnings.warn(f"Optimization status: {problem.status}")
        return np.zeros(d)
    
    return f.value


def mean_variance_kelly(
    mu_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    risk_free: float = 0.0,
    risk_aversion: float = 1.0,
    leverage_limit: float = 1.0,
    long_only: bool = True,
) -> np.ndarray:
    """
    Mean-variance Kelly with risk aversion parameter.
    
    Solves:
        max_f  f'(μ-r) - (γ/2)f'Σf
    
    When γ = 1, this is classical Kelly.
    When γ > 1, this is more conservative.
    
    Parameters
    ----------
    mu_hat : np.ndarray
        Expected returns (d,)
    Sigma_hat : np.ndarray
        Covariance matrix (d, d)
    risk_free : float
        Risk-free rate
    risk_aversion : float
        Risk aversion (1 = Kelly, >1 = conservative)
    leverage_limit : float
        Maximum leverage
    long_only : bool
        Long-only constraint
        
    Returns
    -------
    np.ndarray
        Optimal weights (d,)
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("cvxpy required")
    
    d = len(mu_hat)
    f = cp.Variable(d)
    
    excess = mu_hat - risk_free * np.ones(d)
    
    # Objective: mean - (γ/2) * variance
    objective = f @ excess - (risk_aversion / 2) * cp.quad_form(f, Sigma_hat)
    
    constraints = [cp.sum(f) <= leverage_limit]
    if long_only:
        constraints.append(f >= 0)
    
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve()
    
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        warnings.warn(f"Optimization status: {problem.status}")
        return np.zeros(d)
    
    return f.value


def robust_kelly_moment_ambiguity(
    mu_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    mu_radius: float,
    sigma_radius: float,
    risk_free: float = 0.0,
    leverage_limit: float = 1.0,
    long_only: bool = True,
) -> np.ndarray:
    """
    Robust Kelly under moment ambiguity.
    
    Considers uncertainty in both mean and covariance:
    - Mean: ||μ - μ̂||_2 ≤ δ_μ
    - Covariance: ||Σ - Σ̂||_F ≤ δ_Σ
    
    Parameters
    ----------
    mu_hat : np.ndarray
        Estimated mean
    Sigma_hat : np.ndarray
        Estimated covariance
    mu_radius : float
        Uncertainty in mean (L2 norm)
    sigma_radius : float
        Uncertainty in covariance (Frobenius norm)
    risk_free : float
        Risk-free rate
    leverage_limit : float
        Maximum leverage
    long_only : bool
        Long-only constraint
        
    Returns
    -------
    np.ndarray
        Robust weights
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("cvxpy required")
    
    d = len(mu_hat)
    f = cp.Variable(d)
    
    excess = mu_hat - risk_free * np.ones(d)
    
    # Worst-case mean: minimize f'μ subject to ||μ - μ̂|| ≤ δ
    # Optimal: μ* = μ̂ - δ_μ * f / ||f||
    # f'μ* = f'μ̂ - δ_μ ||f||
    
    # Worst-case variance: maximize f'Σf subject to ||Σ - Σ̂|| ≤ δ
    # Upper bound: f'Σ̂f + δ_Σ ||f||²
    
    # Conservative growth rate
    growth_wc = (
        f @ excess 
        - mu_radius * cp.norm(f, 2)
        - 0.5 * (cp.quad_form(f, Sigma_hat) + sigma_radius * cp.sum_squares(f))
    )
    
    constraints = [cp.sum(f) <= leverage_limit]
    if long_only:
        constraints.append(f >= 0)
    
    problem = cp.Problem(cp.Maximize(growth_wc), constraints)
    problem.solve()
    
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        warnings.warn(f"Optimization status: {problem.status}")
        return np.zeros(d)
    
    return f.value

