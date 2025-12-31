"""
Conformal Prediction for Ambiguity Calibration.

Implements distribution-free calibration methods based on conformal
prediction theory. These methods provide finite-sample coverage
guarantees without distributional assumptions.

Key References:
- Vovk, Gammerman, Shafer (2005): "Algorithmic Learning in a Random World"
- Lei et al. (2018): "Distribution-Free Predictive Inference for Regression"
- Barber et al. (2023): "Conformal Prediction Under Covariate Shift"

Author: Agna Chan
Date: December 2025
"""

import numpy as np
from typing import Tuple, Optional
import warnings


def calibrate_epsilon_conformal(
    returns: np.ndarray,
    alpha: float = 0.1,
    split_ratio: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Calibrate ambiguity radius via split conformal prediction.
    
    This is the main conformal calibration function, providing 
    distribution-free coverage guarantee:
    
        P(|μ̂ - μ| ≤ ε) ≥ 1 - α
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns (n,)
    alpha : float
        Miscoverage rate (e.g., 0.1 for 90% confidence)
    split_ratio : float
        Fraction of data for training (rest for calibration)
        
    Returns
    -------
    epsilon : float
        Calibrated ambiguity radius
    mu_hat : float
        Estimated mean (from training set)
    sigma2_hat : float
        Estimated variance (from training set)
        
    Notes
    -----
    The split conformal method:
    1. Splits data into training and calibration sets
    2. Fits model on training set
    3. Computes residuals on calibration set
    4. Uses quantile of residuals as the prediction interval width
    
    The finite-sample validity comes from exchangeability of the
    calibration residuals with a future residual.
    """
    n = len(returns)
    m = int(n * split_ratio)
    
    if m < 10:
        warnings.warn("Training set too small; results may be unreliable")
    if n - m < 10:
        warnings.warn("Calibration set too small; results may be unreliable")
    
    # Split data
    train = returns[:m]
    calib = returns[m:]
    
    # Fit on training set
    mu_hat = np.mean(train)
    sigma2_hat = np.var(train, ddof=1)
    
    # Nonconformity scores on calibration set
    # Score = |X - μ̂| (absolute residual)
    scores = np.abs(calib - mu_hat)
    
    # Quantile for coverage
    # Use ceiling correction for finite-sample validity:
    # q = ceil((1-α)(n_calib + 1)) / n_calib
    n_calib = len(calib)
    q_level = np.ceil((1 - alpha) * (n_calib + 1)) / n_calib
    q_level = min(q_level, 1.0)  # Cap at 1.0
    
    epsilon = np.quantile(scores, q_level)
    
    return epsilon, mu_hat, sigma2_hat


def split_conformal_calibration(
    returns: np.ndarray,
    alpha: float = 0.1,
    split_ratio: float = 0.5,
    score_fn: Optional[callable] = None,
) -> Tuple[float, dict]:
    """
    Generalized split conformal calibration with custom score function.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    alpha : float
        Miscoverage rate
    split_ratio : float
        Data split ratio
    score_fn : callable, optional
        Custom nonconformity score function.
        Signature: score_fn(data, fitted_params) -> scores
        Default: absolute deviation from mean
        
    Returns
    -------
    epsilon : float
        Calibrated threshold
    diagnostics : dict
        Fitting diagnostics (mu_hat, sigma2_hat, scores, etc.)
    """
    n = len(returns)
    m = int(n * split_ratio)
    
    train = returns[:m]
    calib = returns[m:]
    
    # Fit parameters
    mu_hat = np.mean(train)
    sigma2_hat = np.var(train, ddof=1)
    
    # Compute scores
    if score_fn is None:
        scores = np.abs(calib - mu_hat)
    else:
        scores = score_fn(calib, {'mu': mu_hat, 'sigma2': sigma2_hat})
    
    # Calibrated quantile
    n_calib = len(calib)
    q_level = np.ceil((1 - alpha) * (n_calib + 1)) / n_calib
    q_level = min(q_level, 1.0)
    
    epsilon = np.quantile(scores, q_level)
    
    diagnostics = {
        'mu_hat': mu_hat,
        'sigma2_hat': sigma2_hat,
        'n_train': m,
        'n_calib': n_calib,
        'scores': scores,
        'q_level': q_level,
    }
    
    return epsilon, diagnostics


def cross_conformal_calibration(
    returns: np.ndarray,
    alpha: float = 0.1,
    n_folds: int = 5,
) -> Tuple[float, float, float]:
    """
    Cross-conformal calibration using K-fold cross-validation.
    
    More data-efficient than split conformal by using all data
    for both training and calibration (in different folds).
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    alpha : float
        Miscoverage rate
    n_folds : int
        Number of cross-validation folds
        
    Returns
    -------
    epsilon : float
        Calibrated ambiguity radius
    mu_hat : float
        Mean estimate (from full data)
    sigma2_hat : float
        Variance estimate (from full data)
        
    Notes
    -----
    The cross-conformal method:
    1. Split data into K folds
    2. For each fold, train on K-1 folds, compute score on held-out
    3. Aggregate all scores for quantile computation
    
    This uses more data but has slightly looser theoretical guarantees.
    """
    n = len(returns)
    fold_size = n // n_folds
    
    if fold_size < 5:
        warnings.warn("Folds too small; consider using fewer folds")
    
    # Collect out-of-fold scores
    all_scores = []
    
    for fold in range(n_folds):
        # Define fold boundaries
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < n_folds - 1 else n
        
        # Split into train and validation
        mask = np.ones(n, dtype=bool)
        mask[start_idx:end_idx] = False
        
        train = returns[mask]
        valid = returns[~mask]
        
        # Fit on training
        mu_fold = np.mean(train)
        
        # Score on validation
        scores_fold = np.abs(valid - mu_fold)
        all_scores.extend(scores_fold)
    
    all_scores = np.array(all_scores)
    
    # Compute quantile (looser bound for cross-conformal)
    # Use (1 - α)(1 + 1/K) approximation
    q_level = (1 - alpha) * (1 + 1 / n_folds)
    q_level = min(q_level, 1.0)
    
    epsilon = np.quantile(all_scores, q_level)
    
    # Final estimates from all data
    mu_hat = np.mean(returns)
    sigma2_hat = np.var(returns, ddof=1)
    
    return epsilon, mu_hat, sigma2_hat


def adaptive_conformal_interval(
    returns: np.ndarray,
    alpha: float = 0.1,
    gamma: float = 0.1,
    window_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive Conformal Inference (ACI) for time-varying coverage.
    
    Adjusts the miscoverage level adaptively to maintain target
    coverage over time, even under distribution shift.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns (in order)
    alpha : float
        Target miscoverage rate
    gamma : float
        Learning rate for adaptation (0.01-0.1 typical)
    window_size : int, optional
        Rolling window for estimation (None = expanding)
        
    Returns
    -------
    epsilons : np.ndarray
        Time-varying epsilon values
    coverages : np.ndarray
        Rolling empirical coverage
    alpha_t : np.ndarray
        Adaptive alpha values
        
    Notes
    -----
    ACI (Adaptive Conformal Inference) from Gibbs & Candes (2021):
    - Adjusts α_t based on recent coverage
    - If coverage too high, increase α (tighter intervals)
    - If coverage too low, decrease α (wider intervals)
    
    This is useful for non-stationary return series.
    """
    n = len(returns)
    
    if window_size is None:
        window_size = n  # Expanding window
    
    # Initialize
    alpha_t = np.zeros(n)
    alpha_t[0] = alpha
    
    epsilons = np.zeros(n)
    coverages = np.zeros(n)
    
    # Track recent errors
    errors = []
    
    for t in range(1, n):
        # Current alpha
        current_alpha = alpha_t[t - 1]
        
        # Estimation window
        start = max(0, t - window_size)
        window = returns[start:t]
        
        if len(window) < 10:
            # Not enough data, use simple estimate
            epsilons[t] = np.std(window) * 2 if len(window) > 1 else 0.01
            coverages[t] = 1.0
            alpha_t[t] = alpha
            continue
        
        # Split for conformal
        m = len(window) // 2
        train = window[:m]
        calib = window[m:]
        
        mu_hat = np.mean(train)
        scores = np.abs(calib - mu_hat)
        
        # Quantile with current alpha
        q_level = 1 - current_alpha
        epsilon_t = np.quantile(scores, min(q_level, 1.0))
        epsilons[t] = epsilon_t
        
        # Check if current observation is covered
        error_t = np.abs(returns[t] - mu_hat)
        covered = error_t <= epsilon_t
        errors.append(1 - int(covered))
        
        # Rolling coverage
        recent_errors = errors[-min(50, len(errors)):]
        coverages[t] = 1 - np.mean(recent_errors)
        
        # Adaptive update: α_{t+1} = α_t + γ(err_t - α)
        # This is gradient descent on coverage error
        alpha_t[t] = alpha_t[t - 1] + gamma * ((1 - int(covered)) - alpha)
        alpha_t[t] = np.clip(alpha_t[t], 0.01, 0.5)  # Reasonable bounds
    
    return epsilons, coverages, alpha_t


def conformalized_quantile_regression(
    returns: np.ndarray,
    quantiles: Tuple[float, float] = (0.1, 0.9),
    alpha: float = 0.1,
    split_ratio: float = 0.5,
) -> Tuple[float, float, dict]:
    """
    Conformalized Quantile Regression (CQR) for asymmetric intervals.
    
    Unlike standard conformal (symmetric intervals), CQR can produce
    asymmetric intervals that better capture skewed distributions.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    quantiles : tuple
        Lower and upper quantile levels for base predictor
    alpha : float
        Miscoverage rate
    split_ratio : float
        Data split ratio
        
    Returns
    -------
    lower : float
        Lower interval bound
    upper : float
        Upper interval bound
    diagnostics : dict
        Fitting diagnostics
        
    Notes
    -----
    CQR is particularly useful when returns are skewed, as it
    allows for asymmetric prediction intervals.
    """
    n = len(returns)
    m = int(n * split_ratio)
    
    train = returns[:m]
    calib = returns[m:]
    
    # Fit quantile estimates on training set
    q_lower = np.quantile(train, quantiles[0])
    q_upper = np.quantile(train, quantiles[1])
    
    # CQR score: max(q_lower - X, X - q_upper)
    # This is 0 if X is between quantiles, positive otherwise
    scores = np.maximum(q_lower - calib, calib - q_upper)
    
    # Calibrated quantile
    n_calib = len(calib)
    q_level = np.ceil((1 - alpha) * (n_calib + 1)) / n_calib
    q_level = min(q_level, 1.0)
    
    Q = np.quantile(scores, q_level)
    
    # Conformalized interval
    lower = q_lower - Q
    upper = q_upper + Q
    
    diagnostics = {
        'q_lower': q_lower,
        'q_upper': q_upper,
        'conformity_adjustment': Q,
        'scores': scores,
    }
    
    return lower, upper, diagnostics

