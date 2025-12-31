"""
Performance Metrics and Statistical Tests for Backtesting.

Provides:
- Standard performance metrics (Sharpe, Sortino, etc.)
- Drawdown analysis
- Statistical tests for strategy comparison

Author: Agna Chan
Date: December 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
from scipy import stats
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    # Returns
    total_return: float
    annualized_return: float
    cagr: float  # Compound Annual Growth Rate
    
    # Risk
    annualized_volatility: float
    downside_volatility: float
    
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Drawdown
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int  # In periods
    
    # Distribution
    skewness: float
    kurtosis: float
    var_95: float  # 95% VaR
    cvar_95: float  # 95% CVaR (Expected Shortfall)
    
    # Win/loss
    win_rate: float
    profit_factor: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'cagr': self.cagr,
            'annualized_volatility': self.annualized_volatility,
            'downside_volatility': self.downside_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'avg_drawdown': self.avg_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
        }


def compute_performance_metrics(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    """
    Compute comprehensive performance metrics.
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Period returns (not log returns)
    risk_free_rate : float
        Risk-free rate per period
    periods_per_year : int
        Number of periods per year (252 for daily, 12 for monthly)
        
    Returns
    -------
    PerformanceMetrics
        Comprehensive metrics
    """
    returns = np.asarray(returns)
    n = len(returns)
    
    if n == 0:
        raise ValueError("Empty returns array")
    
    # Basic return metrics
    total_return = np.prod(1 + returns) - 1
    n_years = n / periods_per_year
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    annualized_return = np.mean(returns) * periods_per_year
    
    # Volatility
    annualized_volatility = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    
    # Downside volatility (semi-deviation)
    downside_returns = returns[returns < risk_free_rate]
    if len(downside_returns) > 1:
        downside_volatility = np.std(downside_returns, ddof=1) * np.sqrt(periods_per_year)
    else:
        downside_volatility = 0.0
    
    # Risk-adjusted returns
    excess_return = annualized_return - risk_free_rate * periods_per_year
    sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0
    sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else np.inf
    
    # Drawdown analysis
    wealth = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(wealth)
    drawdowns = (wealth - running_max) / running_max
    max_drawdown = -np.min(drawdowns)
    avg_drawdown = -np.mean(drawdowns[drawdowns < 0]) if np.any(drawdowns < 0) else 0
    
    # Max drawdown duration
    in_drawdown = drawdowns < 0
    if np.any(in_drawdown):
        # Find longest consecutive drawdown
        drawdown_lengths = []
        current_length = 0
        for dd in in_drawdown:
            if dd:
                current_length += 1
            else:
                if current_length > 0:
                    drawdown_lengths.append(current_length)
                current_length = 0
        if current_length > 0:
            drawdown_lengths.append(current_length)
        max_drawdown_duration = max(drawdown_lengths) if drawdown_lengths else 0
    else:
        max_drawdown_duration = 0
    
    calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else np.inf
    
    # Distribution moments
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)  # Excess kurtosis
    
    # Value at Risk and Expected Shortfall
    var_95 = -np.percentile(returns, 5)
    tail_returns = returns[returns <= np.percentile(returns, 5)]
    cvar_95 = -np.mean(tail_returns) if len(tail_returns) > 0 else var_95
    
    # Win/loss statistics
    wins = returns > 0
    win_rate = np.mean(wins)
    
    total_gains = np.sum(returns[returns > 0])
    total_losses = -np.sum(returns[returns < 0])
    profit_factor = total_gains / total_losses if total_losses > 0 else np.inf
    
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        cagr=cagr,
        annualized_volatility=annualized_volatility,
        downside_volatility=downside_volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        avg_drawdown=avg_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        skewness=skewness,
        kurtosis=kurtosis,
        var_95=var_95,
        cvar_95=cvar_95,
        win_rate=win_rate,
        profit_factor=profit_factor,
    )


def compute_drawdown_series(
    returns: Union[pd.Series, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute drawdown time series.
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Period returns
        
    Returns
    -------
    wealth : np.ndarray
        Cumulative wealth
    drawdown : np.ndarray
        Drawdown series (negative values)
    underwater : np.ndarray
        Duration in drawdown at each point
    """
    returns = np.asarray(returns)
    
    wealth = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth - running_max) / running_max
    
    # Time underwater
    underwater = np.zeros_like(drawdown)
    current_underwater = 0
    for i in range(len(drawdown)):
        if drawdown[i] < 0:
            current_underwater += 1
        else:
            current_underwater = 0
        underwater[i] = current_underwater
    
    return wealth, drawdown, underwater


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # At 5% level
    
    def __repr__(self) -> str:
        sig_str = "significant" if self.significant else "not significant"
        return f"{self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.4f} ({sig_str})"


def statistical_tests(
    returns1: Union[pd.Series, np.ndarray],
    returns2: Union[pd.Series, np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, StatisticalTestResult]:
    """
    Perform statistical tests comparing two return series.
    
    Parameters
    ----------
    returns1, returns2 : pd.Series or np.ndarray
        Return series to compare
    alpha : float
        Significance level
        
    Returns
    -------
    Dict[str, StatisticalTestResult]
        Results of various statistical tests
    """
    r1 = np.asarray(returns1)
    r2 = np.asarray(returns2)
    
    results = {}
    
    # === 1. Paired t-test on returns ===
    t_stat, p_value = stats.ttest_rel(r1, r2)
    results['paired_t_test'] = StatisticalTestResult(
        test_name="Paired t-test (H0: mean difference = 0)",
        statistic=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
    )
    
    # === 2. Wilcoxon signed-rank test (non-parametric) ===
    try:
        stat, p_value = stats.wilcoxon(r1 - r2, alternative='two-sided')
        results['wilcoxon'] = StatisticalTestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=stat,
            p_value=p_value,
            significant=p_value < alpha,
        )
    except ValueError:
        # Can fail if all differences are zero
        pass
    
    # === 3. Diebold-Mariano test for predictive accuracy ===
    dm_stat, dm_pvalue = diebold_mariano_test(r1, r2)
    results['diebold_mariano'] = StatisticalTestResult(
        test_name="Diebold-Mariano test",
        statistic=dm_stat,
        p_value=dm_pvalue,
        significant=dm_pvalue < alpha,
    )
    
    # === 4. Sharpe ratio test ===
    sr_stat, sr_pvalue = sharpe_ratio_test(r1, r2)
    results['sharpe_difference'] = StatisticalTestResult(
        test_name="Sharpe ratio difference test",
        statistic=sr_stat,
        p_value=sr_pvalue,
        significant=sr_pvalue < alpha,
    )
    
    return results


def diebold_mariano_test(
    returns1: np.ndarray,
    returns2: np.ndarray,
    h: int = 1,
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for comparing forecast/strategy accuracy.
    
    Tests H0: E[L(e1)] = E[L(e2)] where L is a loss function.
    Here we use L(r) = -r (negative returns as loss).
    
    Parameters
    ----------
    returns1, returns2 : np.ndarray
        Return series
    h : int
        Forecast horizon (for HAC correction)
        
    Returns
    -------
    statistic : float
        DM test statistic
    p_value : float
        Two-sided p-value
    """
    # Loss differential (higher return = lower loss)
    d = returns2 - returns1  # d > 0 means strategy 2 is better
    
    n = len(d)
    d_bar = np.mean(d)
    
    # HAC variance estimator (Newey-West)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    
    for k in range(1, h + 1):
        weight = 1 - k / (h + 1)  # Bartlett kernel
        gamma_k = np.cov(d[k:], d[:-k])[0, 1] if len(d) > k else 0
        gamma_sum += 2 * weight * gamma_k
    
    var_d = (gamma_0 + gamma_sum) / n
    
    if var_d <= 0:
        return 0.0, 1.0
    
    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value


def sharpe_ratio_test(
    returns1: np.ndarray,
    returns2: np.ndarray,
) -> Tuple[float, float]:
    """
    Test for difference in Sharpe ratios.
    
    Uses the Jobson-Korkie test with Memmel correction.
    
    Parameters
    ----------
    returns1, returns2 : np.ndarray
        Return series
        
    Returns
    -------
    statistic : float
        Test statistic
    p_value : float
        Two-sided p-value
    """
    n = len(returns1)
    
    mu1, mu2 = np.mean(returns1), np.mean(returns2)
    sig1, sig2 = np.std(returns1, ddof=1), np.std(returns2, ddof=1)
    
    # Sharpe ratios
    sr1 = mu1 / sig1 if sig1 > 0 else 0
    sr2 = mu2 / sig2 if sig2 > 0 else 0
    
    # Correlation between return series
    rho = np.corrcoef(returns1, returns2)[0, 1]
    
    # Jobson-Korkie variance (with Memmel correction)
    theta = (
        (1 / n) * (
            2 * (1 - rho) + 
            0.5 * (sr1**2 + sr2**2 - 2 * sr1 * sr2 * rho**2)
        )
    )
    
    if theta <= 0:
        return 0.0, 1.0
    
    z_stat = (sr1 - sr2) / np.sqrt(theta)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return z_stat, p_value


def bootstrap_confidence_interval(
    returns: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a metric.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    metric_fn : callable
        Function that computes metric from returns
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level
    seed : int, optional
        Random seed
        
    Returns
    -------
    point_estimate : float
        Point estimate of metric
    lower : float
        Lower confidence bound
    upper : float
        Upper confidence bound
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(returns)
    point_estimate = metric_fn(returns)
    
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        bootstrap_sample = returns[idx]
        bootstrap_estimates.append(metric_fn(bootstrap_sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    
    return point_estimate, lower, upper

