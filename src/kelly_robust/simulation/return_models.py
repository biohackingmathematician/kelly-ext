"""
Return Generation Models for Monte Carlo Simulation.

Implements various return distributions and dynamics:
1. Geometric Brownian Motion (GBM) - baseline
2. Student-t - fat tails
3. GARCH - volatility clustering
4. Jump-diffusion (Merton) - rare events
5. Regime-switching - structural breaks

Author: Agna Chan
Date: December 2025
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum


class ReturnModel(Enum):
    """Enumeration of available return models."""
    GBM = 'gbm'
    STUDENT_T = 't'
    GARCH = 'garch'
    JUMP_DIFFUSION = 'jump'
    REGIME_SWITCHING = 'regime'


# =============================================================================
# 1. Geometric Brownian Motion (Baseline)
# =============================================================================

def simulate_gbm_returns(
    mu: float,
    sigma: float,
    n_periods: int,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate log-returns from Geometric Brownian Motion.
    
    Log-returns: X_t = (μ - σ²/2) + σ·Z_t, where Z_t ~ N(0,1)
    
    Parameters
    ----------
    mu : float
        Drift (expected return per period)
    sigma : float
        Volatility (per period)
    n_periods : int
        Number of time steps
    n_paths : int
        Number of sample paths
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Log-returns, shape (n_paths, n_periods) or (n_periods,) if n_paths=1
        
    Notes
    -----
    Under GBM, the expected log-return is (μ - σ²/2), not μ.
    This is the Itô correction for continuous-time models.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Log-return parameters (Itô correction)
    drift = mu - 0.5 * sigma**2
    
    # Generate standard normal innovations
    Z = np.random.randn(n_paths, n_periods)
    
    # Log-returns
    returns = drift + sigma * Z
    
    return returns.flatten() if n_paths == 1 else returns


# =============================================================================
# 2. Student-t Distributed Returns (Fat Tails)
# =============================================================================

def simulate_t_returns(
    mu: float,
    sigma: float,
    nu: float,
    n_periods: int,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate returns from scaled Student-t distribution.
    
    X_t = μ + σ·√((ν-2)/ν)·T_t, where T_t ~ t(ν)
    
    The scaling ensures Var(X_t) = σ² when ν > 2.
    
    Parameters
    ----------
    mu : float
        Mean return
    sigma : float
        Standard deviation
    nu : float
        Degrees of freedom (higher = more normal, lower = fatter tails)
        Typical values: 3-8 for financial returns
    n_periods : int
        Number of time steps
    n_paths : int
        Number of paths
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Returns with t-distributed innovations
        
    Notes
    -----
    The t-distribution captures the heavy tails observed in actual
    financial returns. For ν ≤ 4, kurtosis is infinite.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if nu <= 2:
        raise ValueError("Degrees of freedom must be > 2 for finite variance")
    
    # Scale to achieve target variance
    scale = sigma * np.sqrt((nu - 2) / nu)
    
    # Generate t-distributed innovations
    T = stats.t.rvs(df=nu, size=(n_paths, n_periods))
    
    # Scaled returns
    returns = mu + scale * T
    
    return returns.flatten() if n_paths == 1 else returns


def estimate_t_parameters(returns: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate Student-t parameters via MLE.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
        
    Returns
    -------
    mu : float
        Location
    sigma : float
        Scale
    nu : float
        Degrees of freedom
    """
    # Use scipy's t.fit
    nu, loc, scale = stats.t.fit(returns)
    
    # Convert to standard parameterization
    sigma = scale * np.sqrt(nu / (nu - 2)) if nu > 2 else scale
    
    return loc, sigma, nu


# =============================================================================
# 3. GARCH(1,1) Model (Volatility Clustering)
# =============================================================================

@dataclass
class GARCHParams:
    """GARCH(1,1) parameters."""
    omega: float  # Constant term
    alpha: float  # ARCH coefficient (lagged squared return)
    beta: float   # GARCH coefficient (lagged variance)
    mu: float = 0.0  # Mean return
    
    @property
    def unconditional_variance(self) -> float:
        """Long-run variance: ω / (1 - α - β)"""
        if self.alpha + self.beta >= 1:
            return np.inf
        return self.omega / (1 - self.alpha - self.beta)
    
    @property
    def persistence(self) -> float:
        """Volatility persistence: α + β"""
        return self.alpha + self.beta


def simulate_garch_returns(
    params: GARCHParams,
    n_periods: int,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate returns from GARCH(1,1) model.
    
    Model:
        r_t = μ + σ_t·Z_t, where Z_t ~ N(0,1)
        σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
    
    Parameters
    ----------
    params : GARCHParams
        GARCH parameters
    n_periods : int
        Number of time steps
    n_paths : int
        Number of paths
    seed : int, optional
        Random seed
        
    Returns
    -------
    returns : np.ndarray
        Simulated returns, shape (n_paths, n_periods)
    volatilities : np.ndarray
        Conditional volatilities, shape (n_paths, n_periods)
        
    Notes
    -----
    GARCH captures the volatility clustering observed in financial
    markets: large moves tend to be followed by large moves.
    """
    if seed is not None:
        np.random.seed(seed)
    
    omega = params.omega
    alpha = params.alpha
    beta = params.beta
    mu = params.mu
    
    # Initialize
    sigma2 = np.full((n_paths, n_periods), params.unconditional_variance)
    returns = np.zeros((n_paths, n_periods))
    
    # Standard normal innovations
    Z = np.random.randn(n_paths, n_periods)
    
    # First period
    returns[:, 0] = mu + np.sqrt(sigma2[:, 0]) * Z[:, 0]
    
    # Simulate forward
    for t in range(1, n_periods):
        sigma2[:, t] = omega + alpha * returns[:, t-1]**2 + beta * sigma2[:, t-1]
        returns[:, t] = mu + np.sqrt(sigma2[:, t]) * Z[:, t]
    
    volatilities = np.sqrt(sigma2)
    
    if n_paths == 1:
        return returns.flatten(), volatilities.flatten()
    return returns, volatilities


def fit_garch(returns: np.ndarray) -> GARCHParams:
    """
    Fit GARCH(1,1) model via quasi-maximum likelihood.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
        
    Returns
    -------
    GARCHParams
        Fitted parameters
    """
    from scipy.optimize import minimize
    
    n = len(returns)
    mu = np.mean(returns)
    demean_returns = returns - mu
    
    def neg_log_likelihood(params):
        omega, alpha, beta = params
        
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(demean_returns)
        
        for t in range(1, n):
            sigma2[t] = omega + alpha * demean_returns[t-1]**2 + beta * sigma2[t-1]
        
        # Avoid numerical issues
        sigma2 = np.maximum(sigma2, 1e-10)
        
        ll = -0.5 * np.sum(np.log(sigma2) + demean_returns**2 / sigma2)
        return -ll
    
    # Initial guess based on unconditional variance
    var0 = np.var(demean_returns)
    x0 = [var0 * 0.1, 0.1, 0.8]
    
    result = minimize(
        neg_log_likelihood, x0,
        bounds=[(1e-8, None), (0, 1), (0, 1)],
        method='L-BFGS-B'
    )
    
    omega, alpha, beta = result.x
    
    return GARCHParams(omega=omega, alpha=alpha, beta=beta, mu=mu)


# =============================================================================
# 4. Jump-Diffusion (Merton Model)
# =============================================================================

@dataclass
class JumpDiffusionParams:
    """Merton jump-diffusion parameters."""
    mu: float         # Drift
    sigma: float      # Diffusion volatility
    lambda_: float    # Jump intensity (jumps per period)
    jump_mean: float  # Mean jump size
    jump_std: float   # Jump size standard deviation


def simulate_jump_diffusion(
    params: JumpDiffusionParams,
    n_periods: int,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate returns from Merton jump-diffusion model.
    
    Model:
        dS/S = μdt + σdW + (J-1)dN
    
    where N is a Poisson process with intensity λ, and J is
    lognormally distributed jump size.
    
    Parameters
    ----------
    params : JumpDiffusionParams
        Model parameters
    n_periods : int
        Number of time steps
    n_paths : int
        Number of paths
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Log-returns including jumps
        
    Notes
    -----
    The jump-diffusion model captures rare but large moves (crashes,
    earnings surprises) that are poorly captured by pure diffusion.
    """
    if seed is not None:
        np.random.seed(seed)
    
    mu = params.mu
    sigma = params.sigma
    lambda_ = params.lambda_
    jump_mean = params.jump_mean
    jump_std = params.jump_std
    
    # Diffusion component (GBM)
    diffusion_drift = mu - 0.5 * sigma**2
    diffusion = diffusion_drift + sigma * np.random.randn(n_paths, n_periods)
    
    # Jump component
    # Number of jumps per period (Poisson)
    n_jumps = np.random.poisson(lambda_, size=(n_paths, n_periods))
    
    # Total jump size per period
    jump_sizes = np.zeros((n_paths, n_periods))
    for i in range(n_paths):
        for t in range(n_periods):
            if n_jumps[i, t] > 0:
                # Sum of log-normal jumps
                jumps = np.random.normal(jump_mean, jump_std, size=n_jumps[i, t])
                jump_sizes[i, t] = np.sum(jumps)
    
    # Compensator for martingale property
    # E[J-1] = exp(jump_mean + jump_std²/2) - 1
    compensator = lambda_ * (np.exp(jump_mean + 0.5 * jump_std**2) - 1)
    
    # Combined return
    returns = diffusion - compensator + jump_sizes
    
    return returns.flatten() if n_paths == 1 else returns


# =============================================================================
# 5. Regime-Switching Model
# =============================================================================

@dataclass
class RegimeSwitchingParams:
    """Two-regime switching parameters."""
    mu_low: float      # Mean in low-vol regime
    sigma_low: float   # Volatility in low-vol regime
    mu_high: float     # Mean in high-vol regime
    sigma_high: float  # Volatility in high-vol regime
    p_low_to_high: float  # Transition probability low → high
    p_high_to_low: float  # Transition probability high → low
    
    @property
    def stationary_prob_low(self) -> float:
        """Stationary probability of being in low-vol regime."""
        return self.p_high_to_low / (self.p_low_to_high + self.p_high_to_low)


def simulate_regime_switching(
    params: RegimeSwitchingParams,
    n_periods: int,
    n_paths: int = 1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate returns from two-regime Markov-switching model.
    
    Model:
        r_t = μ_{s_t} + σ_{s_t}·Z_t
    
    where s_t ∈ {0, 1} follows a Markov chain.
    
    Parameters
    ----------
    params : RegimeSwitchingParams
        Model parameters
    n_periods : int
        Number of periods
    n_paths : int
        Number of paths
    seed : int, optional
        Random seed
        
    Returns
    -------
    returns : np.ndarray
        Simulated returns
    regimes : np.ndarray
        Regime indicators (0 = low vol, 1 = high vol)
        
    Notes
    -----
    Regime-switching models capture structural breaks in market
    conditions (bull/bear markets, crisis/calm periods).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize regime based on stationary distribution
    p_low = params.stationary_prob_low
    regimes = np.zeros((n_paths, n_periods), dtype=int)
    regimes[:, 0] = (np.random.random(n_paths) > p_low).astype(int)
    
    # Transition matrix
    # [P(stay low), P(low→high)]
    # [P(high→low), P(stay high)]
    P = np.array([
        [1 - params.p_low_to_high, params.p_low_to_high],
        [params.p_high_to_low, 1 - params.p_high_to_low]
    ])
    
    # Simulate regimes via Markov chain
    for t in range(1, n_periods):
        for p in range(n_paths):
            current = regimes[p, t-1]
            regimes[p, t] = np.random.choice([0, 1], p=P[current])
    
    # Generate returns based on regime
    mus = np.array([params.mu_low, params.mu_high])
    sigmas = np.array([params.sigma_low, params.sigma_high])
    
    Z = np.random.randn(n_paths, n_periods)
    returns = mus[regimes] + sigmas[regimes] * Z
    
    if n_paths == 1:
        return returns.flatten(), regimes.flatten()
    return returns, regimes


# =============================================================================
# Factory Function
# =============================================================================

def create_return_simulator(
    model: Union[ReturnModel, str],
    **params
) -> Callable[[int, int, Optional[int]], np.ndarray]:
    """
    Factory function to create a return simulator.
    
    Parameters
    ----------
    model : ReturnModel or str
        Model type
    **params
        Model-specific parameters
        
    Returns
    -------
    Callable
        Function with signature (n_periods, n_paths, seed) -> returns
        
    Examples
    --------
    >>> sim = create_return_simulator('gbm', mu=0.0005, sigma=0.02)
    >>> returns = sim(252, 1000, seed=42)
    """
    if isinstance(model, str):
        model = ReturnModel(model.lower())
    
    if model == ReturnModel.GBM:
        mu = params.get('mu', 0.0005)
        sigma = params.get('sigma', 0.02)
        return lambda n, p, s: simulate_gbm_returns(mu, sigma, n, p, s)
    
    elif model == ReturnModel.STUDENT_T:
        mu = params.get('mu', 0.0005)
        sigma = params.get('sigma', 0.02)
        nu = params.get('nu', 5)
        return lambda n, p, s: simulate_t_returns(mu, sigma, nu, n, p, s)
    
    elif model == ReturnModel.GARCH:
        garch_params = params.get('garch_params', GARCHParams(
            omega=1e-6, alpha=0.1, beta=0.85, mu=0.0005
        ))
        return lambda n, p, s: simulate_garch_returns(garch_params, n, p, s)[0]
    
    elif model == ReturnModel.JUMP_DIFFUSION:
        jump_params = params.get('jump_params', JumpDiffusionParams(
            mu=0.0005, sigma=0.015, lambda_=0.1,
            jump_mean=-0.02, jump_std=0.03
        ))
        return lambda n, p, s: simulate_jump_diffusion(jump_params, n, p, s)
    
    elif model == ReturnModel.REGIME_SWITCHING:
        regime_params = params.get('regime_params', RegimeSwitchingParams(
            mu_low=0.001, sigma_low=0.01,
            mu_high=-0.0005, sigma_high=0.03,
            p_low_to_high=0.02, p_high_to_low=0.1
        ))
        return lambda n, p, s: simulate_regime_switching(regime_params, n, p, s)[0]
    
    else:
        raise ValueError(f"Unknown model: {model}")


# =============================================================================
# Utility Functions
# =============================================================================

def compute_return_statistics(returns: np.ndarray) -> dict:
    """
    Compute summary statistics for return series.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series (can be 2D: n_paths × n_periods)
        
    Returns
    -------
    dict
        Summary statistics
    """
    returns = np.atleast_1d(returns).flatten()
    
    return {
        'mean': np.mean(returns),
        'std': np.std(returns, ddof=1),
        'skewness': stats.skew(returns),
        'kurtosis': stats.kurtosis(returns),  # Excess kurtosis
        'min': np.min(returns),
        'max': np.max(returns),
        'median': np.median(returns),
        'q05': np.percentile(returns, 5),
        'q95': np.percentile(returns, 95),
        'jarque_bera': stats.jarque_bera(returns)[0],
        'jb_pvalue': stats.jarque_bera(returns)[1],
    }


def test_normality(returns: np.ndarray) -> dict:
    """
    Test for normality of returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
        
    Returns
    -------
    dict
        Test results
    """
    returns = np.atleast_1d(returns).flatten()
    
    # Jarque-Bera
    jb_stat, jb_pval = stats.jarque_bera(returns)
    
    # Shapiro-Wilk (for smaller samples)
    if len(returns) <= 5000:
        sw_stat, sw_pval = stats.shapiro(returns)
    else:
        sw_stat, sw_pval = np.nan, np.nan
    
    # Kolmogorov-Smirnov
    ks_stat, ks_pval = stats.kstest(
        (returns - np.mean(returns)) / np.std(returns),
        'norm'
    )
    
    return {
        'jarque_bera_stat': jb_stat,
        'jarque_bera_pval': jb_pval,
        'shapiro_wilk_stat': sw_stat,
        'shapiro_wilk_pval': sw_pval,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'is_normal_jb': jb_pval > 0.05,
        'is_normal_ks': ks_pval > 0.05,
    }

