"""
Data Preprocessing Utilities.

Functions for cleaning, transforming, and preparing financial data
for Kelly optimization.

Author: Agna Chan
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Union


def compute_returns(
    prices: pd.DataFrame,
    method: str = 'log',
    fill_method: Optional[str] = 'ffill',
) -> pd.DataFrame:
    """
    Compute returns from price data.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data (T × d)
    method : str
        'log' for log-returns, 'simple' for arithmetic returns
    fill_method : str, optional
        Method to fill missing prices before return computation
        
    Returns
    -------
    pd.DataFrame
        Returns (T-1 × d)
    """
    if fill_method:
        prices = prices.fillna(method=fill_method)
    
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    elif method == 'simple':
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return returns.dropna(how='all')


def align_series(
    *series: pd.DataFrame,
    method: str = 'inner',
) -> Tuple[pd.DataFrame, ...]:
    """
    Align multiple time series to common dates.
    
    Parameters
    ----------
    *series : pd.DataFrame
        DataFrames to align
    method : str
        'inner' (intersection) or 'outer' (union with NaN)
        
    Returns
    -------
    Tuple[pd.DataFrame, ...]
        Aligned DataFrames
    """
    if len(series) == 0:
        return ()
    
    if len(series) == 1:
        return series
    
    # Get common index
    if method == 'inner':
        common_idx = series[0].index
        for s in series[1:]:
            common_idx = common_idx.intersection(s.index)
        
        return tuple(s.loc[common_idx] for s in series)
    
    elif method == 'outer':
        common_idx = series[0].index
        for s in series[1:]:
            common_idx = common_idx.union(s.index)
        
        return tuple(s.reindex(common_idx) for s in series)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def filter_universe(
    prices: pd.DataFrame,
    min_history: int = 252,
    max_missing_pct: float = 0.05,
    min_price: float = 5.0,
    min_volume: Optional[pd.DataFrame] = None,
    min_avg_volume: float = 100000,
) -> pd.DataFrame:
    """
    Filter assets based on data quality criteria.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    min_history : int
        Minimum number of valid observations
    max_missing_pct : float
        Maximum fraction of missing values
    min_price : float
        Minimum average price
    min_volume : pd.DataFrame, optional
        Volume data for liquidity filter
    min_avg_volume : float
        Minimum average daily volume
        
    Returns
    -------
    pd.DataFrame
        Filtered price data
    """
    valid_tickers = []
    
    for col in prices.columns:
        series = prices[col]
        
        # Check history length
        n_valid = series.notna().sum()
        if n_valid < min_history:
            continue
        
        # Check missing percentage
        missing_pct = series.isna().mean()
        if missing_pct > max_missing_pct:
            continue
        
        # Check minimum price
        avg_price = series.mean()
        if avg_price < min_price:
            continue
        
        # Check volume if provided
        if min_volume is not None and col in min_volume.columns:
            avg_vol = min_volume[col].mean()
            if avg_vol < min_avg_volume:
                continue
        
        valid_tickers.append(col)
    
    return prices[valid_tickers]


def compute_rolling_statistics(
    returns: pd.DataFrame,
    window: int = 252,
    min_periods: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute rolling mean, volatility, and Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return data
    window : int
        Rolling window size
    min_periods : int, optional
        Minimum observations for valid estimate
        
    Returns
    -------
    rolling_mean : pd.DataFrame
        Rolling mean returns (annualized)
    rolling_vol : pd.DataFrame
        Rolling volatility (annualized)
    rolling_sharpe : pd.DataFrame
        Rolling Sharpe ratio
    """
    if min_periods is None:
        min_periods = window // 2
    
    # Annualization factor (assuming daily data)
    ann_factor = 252
    
    rolling_mean = returns.rolling(window, min_periods=min_periods).mean() * ann_factor
    rolling_vol = returns.rolling(window, min_periods=min_periods).std() * np.sqrt(ann_factor)
    rolling_sharpe = rolling_mean / rolling_vol
    
    return rolling_mean, rolling_vol, rolling_sharpe


def compute_covariance_matrix(
    returns: pd.DataFrame,
    method: str = 'sample',
    shrinkage_target: str = 'constant_correlation',
    shrinkage_intensity: Optional[float] = None,
) -> np.ndarray:
    """
    Compute covariance matrix with optional shrinkage.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return data
    method : str
        'sample', 'shrinkage', or 'factor'
    shrinkage_target : str
        Target for shrinkage: 'constant_correlation', 'identity', 'diagonal'
    shrinkage_intensity : float, optional
        Shrinkage intensity (auto-computed if None)
        
    Returns
    -------
    np.ndarray
        Covariance matrix (d × d)
    """
    returns = returns.dropna()
    n, d = returns.shape
    
    # Sample covariance
    S = returns.cov().values
    
    if method == 'sample':
        return S
    
    elif method == 'shrinkage':
        # Compute shrinkage target
        if shrinkage_target == 'identity':
            mu = np.trace(S) / d
            F = mu * np.eye(d)
            
        elif shrinkage_target == 'diagonal':
            F = np.diag(np.diag(S))
            
        elif shrinkage_target == 'constant_correlation':
            # Ledoit-Wolf constant correlation
            var = np.diag(S)
            std = np.sqrt(var)
            
            # Average correlation
            corr = S / np.outer(std, std)
            np.fill_diagonal(corr, 0)
            rho_bar = np.sum(corr) / (d * (d - 1))
            
            F = np.outer(std, std) * rho_bar
            np.fill_diagonal(F, var)
        
        else:
            raise ValueError(f"Unknown shrinkage target: {shrinkage_target}")
        
        # Compute shrinkage intensity if not provided
        if shrinkage_intensity is None:
            shrinkage_intensity = _ledoit_wolf_shrinkage(returns.values, S, F)
        
        # Shrunk covariance
        return shrinkage_intensity * F + (1 - shrinkage_intensity) * S
    
    elif method == 'factor':
        # Simple single-factor model
        # R = β × F + ε
        # Σ = β β' σ²_F + D
        market = returns.mean(axis=1)  # Proxy for market factor
        
        betas = np.zeros(d)
        residuals = np.zeros((n, d))
        
        for i in range(d):
            cov_im = np.cov(returns.iloc[:, i], market)[0, 1]
            var_m = np.var(market)
            betas[i] = cov_im / var_m if var_m > 0 else 0
            residuals[:, i] = returns.iloc[:, i] - betas[i] * market
        
        var_market = np.var(market)
        D = np.diag(np.var(residuals, axis=0))
        
        return np.outer(betas, betas) * var_market + D
    
    else:
        raise ValueError(f"Unknown method: {method}")


def _ledoit_wolf_shrinkage(X: np.ndarray, S: np.ndarray, F: np.ndarray) -> float:
    """
    Compute optimal Ledoit-Wolf shrinkage intensity.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (n × d)
    S : np.ndarray
        Sample covariance
    F : np.ndarray
        Shrinkage target
        
    Returns
    -------
    float
        Optimal shrinkage intensity in [0, 1]
    """
    n, d = X.shape
    
    # Center the data
    X = X - X.mean(axis=0)
    
    # Compute the optimal shrinkage intensity
    # Based on Ledoit & Wolf (2004)
    
    # Sum of squared sample eigenvalues
    sum_sq = np.sum(S ** 2)
    
    # Sum of squared off-diagonal elements
    sum_sq_offdiag = sum_sq - np.sum(np.diag(S) ** 2)
    
    # Estimate of variance of off-diagonal elements
    Y = X ** 2
    phi = np.sum(Y.T @ Y) / n - sum_sq
    
    # Compute shrinkage intensity
    delta = S - F
    kappa = np.sum(delta ** 2)
    
    if kappa == 0:
        return 1.0
    
    intensity = phi / (n * kappa)
    
    return np.clip(intensity, 0, 1)


def winsorize_returns(
    returns: pd.DataFrame,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """
    Winsorize extreme returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return data
    lower_pct : float
        Lower percentile cutoff
    upper_pct : float
        Upper percentile cutoff
        
    Returns
    -------
    pd.DataFrame
        Winsorized returns
    """
    lower = returns.quantile(lower_pct)
    upper = returns.quantile(upper_pct)
    
    return returns.clip(lower=lower, upper=upper, axis=1)


def compute_excess_returns(
    returns: pd.DataFrame,
    risk_free: Union[float, pd.Series],
) -> pd.DataFrame:
    """
    Compute excess returns over risk-free rate.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns
    risk_free : float or pd.Series
        Risk-free rate (same frequency as returns)
        
    Returns
    -------
    pd.DataFrame
        Excess returns
    """
    if isinstance(risk_free, (int, float)):
        return returns - risk_free
    else:
        # Align risk-free series
        rf_aligned = risk_free.reindex(returns.index).fillna(method='ffill')
        return returns.sub(rf_aligned, axis=0)


def compute_correlation_matrix(
    returns: pd.DataFrame,
    method: str = 'pearson',
) -> pd.DataFrame:
    """
    Compute correlation matrix.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return data
    method : str
        'pearson', 'spearman', or 'kendall'
        
    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    return returns.corr(method=method)

