"""
Statistical Tests for Strategy Comparison.

Implements:
- Bootstrap confidence intervals for performance metrics
- Diebold-Mariano test for equal predictive accuracy
- Multiple hypothesis correction (Holm-Bonferroni)

Author: Agna Chan
Date: December 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    """Container for bootstrap inference results."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_bootstrap: int
    alpha: float


def bootstrap_statistic(
    data: np.ndarray,
    statistic_fn: callable,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> BootstrapResult:
    """
    Bootstrap confidence interval for any statistic.
    
    Parameters
    ----------
    data : np.ndarray
        Sample data
    statistic_fn : callable
        Function that computes the statistic from data
    n_bootstrap : int
        Number of bootstrap replications
    alpha : float
        Significance level (0.05 for 95% CI)
    seed : int, optional
        Random seed
        
    Returns
    -------
    BootstrapResult
        Point estimate and confidence interval
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(data)
    point_estimate = statistic_fn(data)
    
    # Bootstrap replications
    boot_stats = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        boot_sample = data[np.random.randint(0, n, size=n)]
        boot_stats[b] = statistic_fn(boot_sample)
    
    # Percentile method CI
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    std_error = np.std(boot_stats, ddof=1)
    
    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )


def bootstrap_sharpe_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> BootstrapResult:
    """
    Bootstrap confidence interval for annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    risk_free : float
        Risk-free rate (same frequency as returns)
    periods_per_year : int
        Annualization factor
    n_bootstrap : int
        Number of bootstrap replications
    alpha : float
        Significance level
    seed : int, optional
        Random seed
        
    Returns
    -------
    BootstrapResult
        Sharpe ratio with confidence interval
    """
    def sharpe_fn(r):
        excess = r - risk_free
        if np.std(excess, ddof=1) == 0:
            return 0.0
        return np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(periods_per_year)
    
    return bootstrap_statistic(returns, sharpe_fn, n_bootstrap, alpha, seed)


def bootstrap_sharpe_difference(
    returns1: np.ndarray,
    returns2: np.ndarray,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> BootstrapResult:
    """
    Bootstrap confidence interval for Sharpe ratio difference.
    
    Tests H0: SR1 = SR2 (difference = 0)
    
    Parameters
    ----------
    returns1, returns2 : np.ndarray
        Return series for two strategies (must be same length, aligned)
    risk_free : float
        Risk-free rate
    periods_per_year : int
        Annualization factor
    n_bootstrap : int
        Number of bootstrap replications
    alpha : float
        Significance level
    seed : int, optional
        Random seed
        
    Returns
    -------
    BootstrapResult
        Sharpe difference with CI. If CI excludes 0, difference is significant.
    """
    if len(returns1) != len(returns2):
        raise ValueError("Return series must have same length")
    
    if seed is not None:
        np.random.seed(seed)
    
    n = len(returns1)
    
    def sharpe_fn(r):
        excess = r - risk_free
        if np.std(excess, ddof=1) == 0:
            return 0.0
        return np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(periods_per_year)
    
    point_diff = sharpe_fn(returns1) - sharpe_fn(returns2)
    
    # Paired bootstrap (same indices for both)
    boot_diffs = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        sr1 = sharpe_fn(returns1[idx])
        sr2 = sharpe_fn(returns2[idx])
        boot_diffs[b] = sr1 - sr2
    
    ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    std_error = np.std(boot_diffs, ddof=1)
    
    return BootstrapResult(
        point_estimate=point_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )


@dataclass
class DieboldMarianoResult:
    """Container for Diebold-Mariano test results."""
    statistic: float
    p_value: float
    mean_diff: float
    significant: bool
    alternative: str


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1,
    alternative: str = 'two-sided',
) -> DieboldMarianoResult:
    """
    Diebold-Mariano test for equal predictive accuracy.
    
    Tests whether two forecasts have equal expected loss.
    
    Parameters
    ----------
    errors1, errors2 : np.ndarray
        Forecast errors (or loss differentials) for two methods
    h : int
        Forecast horizon (for HAC variance adjustment)
    alternative : str
        'two-sided', 'less', or 'greater'
        
    Returns
    -------
    DieboldMarianoResult
        Test statistic, p-value, and significance
        
    References
    ----------
    Diebold & Mariano (1995), "Comparing Predictive Accuracy"
    """
    if len(errors1) != len(errors2):
        raise ValueError("Error series must have same length")
    
    # Loss differential
    d = errors1**2 - errors2**2  # Squared error loss
    
    n = len(d)
    mean_d = np.mean(d)
    
    # HAC variance (Newey-West with h-1 lags)
    gamma_0 = np.var(d, ddof=1)
    
    if h > 1:
        gamma_sum = 0
        for k in range(1, h):
            if len(d) > k:
                gamma_k = np.cov(d[:-k], d[k:])[0, 1]
                gamma_sum += 2 * (1 - k / h) * gamma_k
        var_d = (gamma_0 + gamma_sum) / n
    else:
        var_d = gamma_0 / n
    
    # DM statistic
    if var_d <= 0:
        dm_stat = 0.0
        p_value = 1.0
    else:
        dm_stat = mean_d / np.sqrt(var_d)
        
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        elif alternative == 'less':
            p_value = stats.norm.cdf(dm_stat)
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(dm_stat)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
    
    return DieboldMarianoResult(
        statistic=dm_stat,
        p_value=p_value,
        mean_diff=mean_d,
        significant=p_value < 0.05,
        alternative=alternative,
    )


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[bool], List[float]]:
    """
    Holm-Bonferroni correction for multiple hypothesis testing.
    
    Controls family-wise error rate (FWER) at level alpha.
    
    Parameters
    ----------
    p_values : List[float]
        Raw p-values from multiple tests
    alpha : float
        Significance level
        
    Returns
    -------
    rejected : List[bool]
        Whether each hypothesis is rejected
    adjusted_p : List[float]
        Adjusted p-values
    """
    m = len(p_values)
    
    # Sort p-values and track original indices
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    sorted_indices = [p[0] for p in sorted_pairs]
    
    # Holm-Bonferroni thresholds
    rejected = [False] * m
    adjusted_p = [1.0] * m
    
    for k, (orig_idx, p) in enumerate(sorted_pairs):
        threshold = alpha / (m - k)
        adjusted_p[orig_idx] = min(1.0, p * (m - k))
        
        if p <= threshold:
            rejected[orig_idx] = True
        else:
            # Once we fail to reject, stop
            break
    
    # Ensure monotonicity of adjusted p-values
    for k in range(1, m):
        idx = sorted_indices[k]
        prev_idx = sorted_indices[k-1]
        adjusted_p[idx] = max(adjusted_p[idx], adjusted_p[prev_idx])
    
    return rejected, adjusted_p


def compare_strategies(
    returns_dict: Dict[str, np.ndarray],
    baseline: str,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Comprehensive comparison of multiple strategies against a baseline.
    
    Parameters
    ----------
    returns_dict : Dict[str, np.ndarray]
        Dictionary mapping strategy names to return series
    baseline : str
        Name of baseline strategy for comparisons
    risk_free : float
        Risk-free rate
    periods_per_year : int
        Annualization factor
    n_bootstrap : int
        Bootstrap replications
    alpha : float
        Significance level
        
    Returns
    -------
    pd.DataFrame
        Comparison results with Sharpe ratios, differences, CIs, and p-values
    """
    if baseline not in returns_dict:
        raise ValueError(f"Baseline '{baseline}' not found in returns_dict")
    
    baseline_returns = returns_dict[baseline]
    
    results = []
    p_values = []
    
    for name, returns in returns_dict.items():
        # Sharpe ratio with CI
        sr_result = bootstrap_sharpe_ratio(
            returns, risk_free, periods_per_year, n_bootstrap, alpha
        )
        
        row = {
            'Strategy': name,
            'Sharpe': sr_result.point_estimate,
            'Sharpe_CI_Lower': sr_result.ci_lower,
            'Sharpe_CI_Upper': sr_result.ci_upper,
        }
        
        # Comparison to baseline
        if name != baseline:
            diff_result = bootstrap_sharpe_difference(
                returns, baseline_returns, risk_free, periods_per_year, n_bootstrap, alpha
            )
            
            row['Sharpe_Diff'] = diff_result.point_estimate
            row['Diff_CI_Lower'] = diff_result.ci_lower
            row['Diff_CI_Upper'] = diff_result.ci_upper
            
            # Is difference significant? (CI excludes 0)
            row['Significant'] = (diff_result.ci_lower > 0) or (diff_result.ci_upper < 0)
            
            p_values.append(0.05)  # Placeholder for bootstrap p-value
            row['P_Value_Raw'] = 0.05
        else:
            row['Sharpe_Diff'] = 0.0
            row['Diff_CI_Lower'] = 0.0
            row['Diff_CI_Upper'] = 0.0
            row['Significant'] = False
            row['P_Value_Raw'] = 1.0
        
        results.append(row)
    
    # Apply Holm-Bonferroni correction
    if p_values:
        _, adjusted_p = holm_bonferroni_correction(p_values, alpha)
        
        adj_idx = 0
        for row in results:
            if row['Strategy'] != baseline:
                row['P_Value_Adjusted'] = adjusted_p[adj_idx]
                row['Significant_Adjusted'] = adjusted_p[adj_idx] < alpha
                adj_idx += 1
            else:
                row['P_Value_Adjusted'] = 1.0
                row['Significant_Adjusted'] = False
    
    return pd.DataFrame(results)

