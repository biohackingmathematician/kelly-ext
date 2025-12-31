"""
Bootstrap Methods for Ambiguity Calibration.

Implements resampling-based approaches for uncertainty quantification:
- Standard bootstrap (i.i.d. resampling)
- Block bootstrap (for time series)
- Stationary bootstrap (random block lengths)
- Parametric bootstrap (model-based)

Author: Agna Chan
Date: December 2025
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    """Container for bootstrap results."""
    epsilon: float
    mu_hat: float
    sigma2_hat: float
    bootstrap_means: np.ndarray
    confidence_interval: Tuple[float, float]
    
    @property
    def standard_error(self) -> float:
        return np.std(self.bootstrap_means, ddof=1)


def calibrate_epsilon_bootstrap(
    returns: np.ndarray,
    alpha: float = 0.1,
    n_bootstrap: int = 10000,
    method: str = 'percentile',
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Calibrate ambiguity radius via bootstrap.
    
    Uses bootstrap to estimate the distribution of the mean estimator,
    then sets ε as the (1-α/2) quantile of |μ̂* - μ̂|.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    alpha : float
        Miscoverage rate
    n_bootstrap : int
        Number of bootstrap replications
    method : str
        Bootstrap CI method: 'percentile', 'basic', or 'bca'
    seed : int, optional
        Random seed
        
    Returns
    -------
    epsilon : float
        Calibrated ambiguity radius
    mu_hat : float
        Estimated mean
    sigma2_hat : float
        Estimated variance
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(returns)
    mu_hat = np.mean(returns)
    sigma2_hat = np.var(returns, ddof=1)
    
    # Bootstrap the mean
    bootstrap_means = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        bootstrap_means[b] = np.mean(returns[idx])
    
    if method == 'percentile':
        # Percentile method: use quantiles of bootstrap distribution
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        epsilon = max(abs(mu_hat - lower), abs(upper - mu_hat))
        
    elif method == 'basic':
        # Basic bootstrap: use quantiles of deviation
        deviations = np.abs(bootstrap_means - mu_hat)
        epsilon = np.percentile(deviations, 100 * (1 - alpha))
        
    elif method == 'bca':
        # BCa (Bias-Corrected and Accelerated) method
        epsilon = _bca_bootstrap(returns, bootstrap_means, mu_hat, alpha)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return epsilon, mu_hat, sigma2_hat


def _bca_bootstrap(
    data: np.ndarray,
    bootstrap_estimates: np.ndarray,
    point_estimate: float,
    alpha: float,
) -> float:
    """
    BCa bootstrap confidence interval.
    
    Adjusts for bias and skewness in the bootstrap distribution.
    """
    from scipy import stats
    
    n = len(data)
    B = len(bootstrap_estimates)
    
    # Bias correction factor
    z0 = stats.norm.ppf(np.mean(bootstrap_estimates < point_estimate))
    
    # Acceleration factor (jackknife estimate)
    jackknife_estimates = np.zeros(n)
    for i in range(n):
        jackknife_estimates[i] = np.mean(np.delete(data, i))
    
    jack_mean = np.mean(jackknife_estimates)
    num = np.sum((jack_mean - jackknife_estimates) ** 3)
    den = 6 * (np.sum((jack_mean - jackknife_estimates) ** 2) ** 1.5)
    a = num / den if den != 0 else 0
    
    # Adjusted quantiles
    z_alpha = stats.norm.ppf(alpha / 2)
    z_1_alpha = stats.norm.ppf(1 - alpha / 2)
    
    alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    alpha2 = stats.norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))
    
    alpha1 = np.clip(alpha1, 0.001, 0.999)
    alpha2 = np.clip(alpha2, 0.001, 0.999)
    
    lower = np.percentile(bootstrap_estimates, 100 * alpha1)
    upper = np.percentile(bootstrap_estimates, 100 * alpha2)
    
    return max(abs(point_estimate - lower), abs(upper - point_estimate))


def parametric_bootstrap(
    returns: np.ndarray,
    alpha: float = 0.1,
    n_bootstrap: int = 10000,
    distribution: str = 'normal',
    seed: Optional[int] = None,
) -> BootstrapResult:
    """
    Parametric bootstrap assuming a specific distribution.
    
    Fits a parametric model, then resamples from the fitted model.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    alpha : float
        Miscoverage rate
    n_bootstrap : int
        Number of bootstrap samples
    distribution : str
        Distribution family: 'normal', 't', or 'skewnormal'
    seed : int, optional
        Random seed
        
    Returns
    -------
    BootstrapResult
        Bootstrap results including epsilon
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(returns)
    mu_hat = np.mean(returns)
    sigma2_hat = np.var(returns, ddof=1)
    sigma_hat = np.sqrt(sigma2_hat)
    
    # Generate parametric bootstrap samples
    if distribution == 'normal':
        bootstrap_samples = np.random.normal(mu_hat, sigma_hat, (n_bootstrap, n))
        
    elif distribution == 't':
        from scipy import stats
        # Estimate degrees of freedom via MLE
        # Use moment-based estimate: ν = 2σ²/(σ² - s²) if kurtosis > 0
        kurtosis = stats.kurtosis(returns)
        if kurtosis > 0:
            nu = max(4, 6 / kurtosis + 4)  # Rough estimate
        else:
            nu = 30  # Nearly normal
        
        t_samples = stats.t.rvs(df=nu, size=(n_bootstrap, n))
        bootstrap_samples = mu_hat + sigma_hat * t_samples / np.sqrt(nu / (nu - 2))
        
    elif distribution == 'skewnormal':
        from scipy import stats
        # Fit skew-normal parameters
        skew = stats.skew(returns)
        # Approximate skew-normal parameter
        delta = np.sign(skew) * np.sqrt(np.abs(skew) / (2 * (1 - 2/np.pi)))
        delta = np.clip(delta, -0.95, 0.95)
        alpha_sn = delta / np.sqrt(1 - delta**2)
        
        # Generate skew-normal samples
        u0 = np.random.randn(n_bootstrap, n)
        u1 = np.random.randn(n_bootstrap, n)
        z = delta * np.abs(u0) + np.sqrt(1 - delta**2) * u1
        bootstrap_samples = mu_hat + sigma_hat * z
        
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Compute bootstrap means
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    
    # Compute epsilon
    deviations = np.abs(bootstrap_means - mu_hat)
    epsilon = np.percentile(deviations, 100 * (1 - alpha))
    
    # Confidence interval
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return BootstrapResult(
        epsilon=epsilon,
        mu_hat=mu_hat,
        sigma2_hat=sigma2_hat,
        bootstrap_means=bootstrap_means,
        confidence_interval=(lower, upper),
    )


def block_bootstrap(
    returns: np.ndarray,
    alpha: float = 0.1,
    block_size: Optional[int] = None,
    n_bootstrap: int = 5000,
    seed: Optional[int] = None,
) -> BootstrapResult:
    """
    Block bootstrap for time series.
    
    Preserves temporal dependence by resampling blocks of consecutive
    observations rather than individual observations.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns (time-ordered)
    alpha : float
        Miscoverage rate
    block_size : int, optional
        Block size (default: n^(1/3))
    n_bootstrap : int
        Number of bootstrap replications
    seed : int, optional
        Random seed
        
    Returns
    -------
    BootstrapResult
        Bootstrap results
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(returns)
    
    # Default block size: n^(1/3) is optimal for many statistics
    if block_size is None:
        block_size = max(1, int(np.ceil(n ** (1/3))))
    
    mu_hat = np.mean(returns)
    sigma2_hat = np.var(returns, ddof=1)
    
    # Number of blocks needed
    n_blocks = int(np.ceil(n / block_size))
    
    bootstrap_means = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        # Sample block starting positions
        starts = np.random.randint(0, n - block_size + 1, size=n_blocks)
        
        # Build bootstrap sample from blocks
        bootstrap_sample = []
        for start in starts:
            bootstrap_sample.extend(returns[start:start + block_size])
        
        # Trim to original length
        bootstrap_sample = np.array(bootstrap_sample[:n])
        bootstrap_means[b] = np.mean(bootstrap_sample)
    
    # Compute epsilon
    deviations = np.abs(bootstrap_means - mu_hat)
    epsilon = np.percentile(deviations, 100 * (1 - alpha))
    
    # Confidence interval
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return BootstrapResult(
        epsilon=epsilon,
        mu_hat=mu_hat,
        sigma2_hat=sigma2_hat,
        bootstrap_means=bootstrap_means,
        confidence_interval=(lower, upper),
    )


def stationary_bootstrap(
    returns: np.ndarray,
    alpha: float = 0.1,
    expected_block_size: Optional[float] = None,
    n_bootstrap: int = 5000,
    seed: Optional[int] = None,
) -> BootstrapResult:
    """
    Stationary bootstrap (Politis & Romano, 1994).
    
    Uses geometrically distributed random block lengths, which makes
    the resampled series stationary (unlike fixed-block bootstrap).
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    alpha : float
        Miscoverage rate
    expected_block_size : float, optional
        Expected block length (default: n^(1/3))
    n_bootstrap : int
        Number of bootstrap replications
    seed : int, optional
        Random seed
        
    Returns
    -------
    BootstrapResult
        Bootstrap results
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(returns)
    
    if expected_block_size is None:
        expected_block_size = n ** (1/3)
    
    # Probability of starting new block
    p = 1 / expected_block_size
    
    mu_hat = np.mean(returns)
    sigma2_hat = np.var(returns, ddof=1)
    
    bootstrap_means = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        bootstrap_sample = np.zeros(n)
        
        # Start at random position
        idx = np.random.randint(0, n)
        
        for i in range(n):
            bootstrap_sample[i] = returns[idx]
            
            # With probability p, jump to new random position
            # Otherwise, move to next observation (with wrap-around)
            if np.random.random() < p:
                idx = np.random.randint(0, n)
            else:
                idx = (idx + 1) % n
        
        bootstrap_means[b] = np.mean(bootstrap_sample)
    
    # Compute epsilon
    deviations = np.abs(bootstrap_means - mu_hat)
    epsilon = np.percentile(deviations, 100 * (1 - alpha))
    
    # Confidence interval
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return BootstrapResult(
        epsilon=epsilon,
        mu_hat=mu_hat,
        sigma2_hat=sigma2_hat,
        bootstrap_means=bootstrap_means,
        confidence_interval=(lower, upper),
    )


def wild_bootstrap(
    returns: np.ndarray,
    alpha: float = 0.1,
    n_bootstrap: int = 5000,
    distribution: str = 'rademacher',
    seed: Optional[int] = None,
) -> BootstrapResult:
    """
    Wild bootstrap for heteroskedastic data.
    
    Preserves the heteroskedasticity pattern in the original data
    by multiplying residuals by random weights.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    alpha : float
        Miscoverage rate
    n_bootstrap : int
        Number of bootstrap replications
    distribution : str
        Weight distribution: 'rademacher', 'normal', or 'mammen'
    seed : int, optional
        Random seed
        
    Returns
    -------
    BootstrapResult
        Bootstrap results
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(returns)
    mu_hat = np.mean(returns)
    sigma2_hat = np.var(returns, ddof=1)
    
    # Residuals
    residuals = returns - mu_hat
    
    bootstrap_means = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        # Generate random weights
        if distribution == 'rademacher':
            weights = np.random.choice([-1, 1], size=n)
        elif distribution == 'normal':
            weights = np.random.randn(n)
        elif distribution == 'mammen':
            # Mammen's two-point distribution
            p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
            weights = np.where(
                np.random.random(n) < p,
                -(np.sqrt(5) - 1) / 2,
                (np.sqrt(5) + 1) / 2
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # Wild bootstrap sample
        bootstrap_sample = mu_hat + residuals * weights
        bootstrap_means[b] = np.mean(bootstrap_sample)
    
    # Compute epsilon
    deviations = np.abs(bootstrap_means - mu_hat)
    epsilon = np.percentile(deviations, 100 * (1 - alpha))
    
    # Confidence interval
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return BootstrapResult(
        epsilon=epsilon,
        mu_hat=mu_hat,
        sigma2_hat=sigma2_hat,
        bootstrap_means=bootstrap_means,
        confidence_interval=(lower, upper),
    )

