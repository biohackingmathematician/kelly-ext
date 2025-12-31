"""
Bayesian Methods for Ambiguity Calibration.

Implements Bayesian approaches for uncertainty quantification:
- Normal-Gamma conjugate prior for Gaussian data
- Posterior credible intervals for the mean
- Posterior-averaged Kelly fractions

Author: Agna Chan
Date: December 2025
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class BayesianResult:
    """Container for Bayesian calibration results."""
    epsilon: float
    mu_posterior_mean: float
    mu_posterior_std: float
    sigma2_posterior_mean: float
    credible_interval: Tuple[float, float]
    posterior_kelly_mean: float
    posterior_kelly_std: float


@dataclass
class NormalGammaPrior:
    """
    Normal-Gamma conjugate prior for (μ, τ) where τ = 1/σ².
    
    Prior specification:
    - μ | τ ~ N(μ_0, 1/(κ_0 * τ))
    - τ ~ Gamma(α_0, β_0)
    
    Parameters
    ----------
    mu_0 : float
        Prior mean for μ
    kappa_0 : float
        Prior precision weight (strength of prior on mean)
    alpha_0 : float
        Gamma shape parameter for τ
    beta_0 : float
        Gamma rate parameter for τ
    """
    mu_0: float = 0.0
    kappa_0: float = 1.0
    alpha_0: float = 1.0
    beta_0: float = 0.001


def normal_gamma_posterior(
    returns: np.ndarray,
    prior: Optional[NormalGammaPrior] = None,
) -> Tuple[float, float, float, float]:
    """
    Compute Normal-Gamma posterior parameters.
    
    Given prior (μ_0, κ_0, α_0, β_0) and data, returns posterior
    parameters (μ_n, κ_n, α_n, β_n).
    
    Parameters
    ----------
    returns : np.ndarray
        Observed returns
    prior : NormalGammaPrior, optional
        Prior specification (default: weakly informative)
        
    Returns
    -------
    mu_n : float
        Posterior mean location
    kappa_n : float
        Posterior precision weight
    alpha_n : float
        Posterior Gamma shape
    beta_n : float
        Posterior Gamma rate
    """
    if prior is None:
        prior = NormalGammaPrior()
    
    n = len(returns)
    x_bar = np.mean(returns)
    s2 = np.var(returns, ddof=0)  # MLE variance
    
    # Posterior updates (conjugate formulas)
    kappa_n = prior.kappa_0 + n
    mu_n = (prior.kappa_0 * prior.mu_0 + n * x_bar) / kappa_n
    alpha_n = prior.alpha_0 + n / 2
    beta_n = (
        prior.beta_0 + 
        0.5 * n * s2 + 
        0.5 * (prior.kappa_0 * n / kappa_n) * (x_bar - prior.mu_0) ** 2
    )
    
    return mu_n, kappa_n, alpha_n, beta_n


def calibrate_epsilon_bayesian(
    returns: np.ndarray,
    alpha: float = 0.1,
    prior: Optional[NormalGammaPrior] = None,
    n_samples: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Calibrate ambiguity radius via Bayesian posterior.
    
    Uses the posterior distribution of μ to construct a credible
    interval, then sets ε as half the interval width.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    alpha : float
        Credible level (1 - α credible interval)
    prior : NormalGammaPrior, optional
        Prior specification
    n_samples : int
        Number of posterior samples
    seed : int, optional
        Random seed
        
    Returns
    -------
    epsilon : float
        Calibrated ambiguity radius (half credible interval width)
    mu_posterior_mean : float
        Posterior mean of μ
    sigma2_posterior_mean : float
        Posterior mean of σ²
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get posterior parameters
    mu_n, kappa_n, alpha_n, beta_n = normal_gamma_posterior(returns, prior)
    
    # Sample from posterior
    # τ ~ Gamma(α_n, β_n)
    tau_samples = np.random.gamma(alpha_n, 1/beta_n, size=n_samples)
    
    # μ | τ ~ N(μ_n, 1/(κ_n * τ))
    mu_samples = mu_n + np.random.randn(n_samples) / np.sqrt(kappa_n * tau_samples)
    
    # Posterior means
    mu_posterior_mean = np.mean(mu_samples)
    sigma2_posterior_mean = np.mean(1 / tau_samples)
    
    # Credible interval for μ
    lower = np.percentile(mu_samples, 100 * alpha / 2)
    upper = np.percentile(mu_samples, 100 * (1 - alpha / 2))
    
    # Epsilon is the maximum deviation from posterior mean
    epsilon = max(abs(mu_posterior_mean - lower), abs(upper - mu_posterior_mean))
    
    return epsilon, mu_posterior_mean, sigma2_posterior_mean


def compute_posterior_kelly(
    returns: np.ndarray,
    risk_free: float = 0.0,
    prior: Optional[NormalGammaPrior] = None,
    n_samples: int = 10000,
    seed: Optional[int] = None,
) -> BayesianResult:
    """
    Compute posterior distribution of the Kelly fraction.
    
    Samples from the joint posterior of (μ, σ²) and computes the
    implied distribution of Kelly fractions.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    risk_free : float
        Risk-free rate
    prior : NormalGammaPrior, optional
        Prior specification
    n_samples : int
        Number of posterior samples
    seed : int, optional
        Random seed
        
    Returns
    -------
    BayesianResult
        Complete Bayesian analysis results
        
    Notes
    -----
    The posterior mean Kelly fraction is the Bayes-optimal allocation
    under squared error loss. However, for growth-rate optimization,
    one might prefer the median or a robust estimator.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get posterior parameters
    mu_n, kappa_n, alpha_n, beta_n = normal_gamma_posterior(returns, prior)
    
    # Sample from posterior
    tau_samples = np.random.gamma(alpha_n, 1/beta_n, size=n_samples)
    sigma2_samples = 1 / tau_samples
    mu_samples = mu_n + np.random.randn(n_samples) / np.sqrt(kappa_n * tau_samples)
    
    # Compute Kelly fractions for each sample
    kelly_samples = (mu_samples - risk_free) / sigma2_samples
    
    # Clip to reasonable range
    kelly_samples = np.clip(kelly_samples, -10, 10)
    
    # Summary statistics
    mu_posterior_mean = np.mean(mu_samples)
    mu_posterior_std = np.std(mu_samples)
    sigma2_posterior_mean = np.mean(sigma2_samples)
    
    # Credible interval for mean
    alpha = 0.1  # 90% credible
    lower = np.percentile(mu_samples, 5)
    upper = np.percentile(mu_samples, 95)
    epsilon = max(abs(mu_posterior_mean - lower), abs(upper - mu_posterior_mean))
    
    return BayesianResult(
        epsilon=epsilon,
        mu_posterior_mean=mu_posterior_mean,
        mu_posterior_std=mu_posterior_std,
        sigma2_posterior_mean=sigma2_posterior_mean,
        credible_interval=(lower, upper),
        posterior_kelly_mean=np.mean(kelly_samples),
        posterior_kelly_std=np.std(kelly_samples),
    )


def bayesian_kelly_robust(
    returns: np.ndarray,
    risk_free: float = 0.0,
    prior: Optional[NormalGammaPrior] = None,
    quantile: float = 0.1,
    n_samples: int = 10000,
    seed: Optional[int] = None,
) -> float:
    """
    Compute robust Bayesian Kelly fraction.
    
    Instead of using the posterior mean of the Kelly fraction,
    uses a lower quantile to be conservative (robust to parameter
    uncertainty).
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    risk_free : float
        Risk-free rate
    prior : NormalGammaPrior, optional
        Prior specification
    quantile : float
        Lower quantile to use (e.g., 0.1 for 10th percentile)
    n_samples : int
        Number of posterior samples
    seed : int, optional
        Random seed
        
    Returns
    -------
    float
        Robust Kelly fraction (lower quantile of posterior)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get posterior parameters
    mu_n, kappa_n, alpha_n, beta_n = normal_gamma_posterior(returns, prior)
    
    # Sample from posterior
    tau_samples = np.random.gamma(alpha_n, 1/beta_n, size=n_samples)
    sigma2_samples = 1 / tau_samples
    mu_samples = mu_n + np.random.randn(n_samples) / np.sqrt(kappa_n * tau_samples)
    
    # Compute Kelly fractions
    kelly_samples = (mu_samples - risk_free) / sigma2_samples
    
    # Return lower quantile (conservative)
    return max(0, np.percentile(kelly_samples, 100 * quantile))


def empirical_bayes_prior(
    returns: np.ndarray,
    n_assets: int = 1,
) -> NormalGammaPrior:
    """
    Construct empirical Bayes prior from data.
    
    Uses the data to inform the prior hyperparameters, useful when
    we have multiple assets or a long history.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    n_assets : int
        Number of assets (for shrinkage strength)
        
    Returns
    -------
    NormalGammaPrior
        Data-informed prior
    """
    n = len(returns)
    
    # Empirical estimates
    mu_hat = np.mean(returns)
    sigma2_hat = np.var(returns, ddof=1)
    
    # Set prior mean to empirical mean
    # But with weak precision (let data dominate)
    mu_0 = mu_hat
    
    # Prior precision: fraction of sample size
    # More shrinkage for more assets (Stein-type)
    kappa_0 = max(1.0, n_assets / 10)
    
    # Gamma parameters from empirical variance
    # Mode of Gamma(α, β) is (α-1)/β if α > 1
    # Mean is α/β
    # Set so posterior is centered near empirical variance
    alpha_0 = 3.0
    beta_0 = 2 * sigma2_hat  # Mode at sigma2_hat/2
    
    return NormalGammaPrior(
        mu_0=mu_0,
        kappa_0=kappa_0,
        alpha_0=alpha_0,
        beta_0=beta_0,
    )


def jeffreys_prior() -> NormalGammaPrior:
    """
    Jeffreys (non-informative) prior for Normal-Gamma.
    
    The Jeffreys prior for (μ, σ²) is proportional to 1/σ²,
    which corresponds to κ_0 → 0, α_0 = 0, β_0 = 0.
    
    We use limiting approximation with small values.
    
    Returns
    -------
    NormalGammaPrior
        Non-informative Jeffreys prior
    """
    return NormalGammaPrior(
        mu_0=0.0,
        kappa_0=0.001,  # Very weak prior on mean
        alpha_0=0.001,  # Nearly flat prior on variance
        beta_0=0.001,
    )

