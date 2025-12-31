"""
Walk-Forward Backtesting Framework for Kelly Strategies.

Implements a rigorous walk-forward testing protocol with:
- Rolling estimation windows
- Non-overlapping out-of-sample periods
- Transaction cost integration
- Multiple strategy comparison

Author: Agna Chan
Date: December 2025
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union, Any
from enum import Enum
import warnings


@dataclass
class BacktestResult:
    """Container for backtest results."""
    
    # Wealth and returns
    wealth_series: pd.Series
    returns_series: pd.Series
    
    # Allocations
    weights_history: pd.DataFrame
    
    # Performance metrics
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trading statistics
    total_turnover: float
    total_costs: float
    num_rebalances: int
    
    # Additional diagnostics
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"BacktestResult(\n"
            f"  total_return={self.total_return:.2%},\n"
            f"  sharpe_ratio={self.sharpe_ratio:.3f},\n"
            f"  max_drawdown={self.max_drawdown:.2%},\n"
            f"  turnover={self.total_turnover:.2f}x\n"
            f")"
        )


class WalkForwardBacktest:
    """
    Walk-forward backtesting engine.
    
    Implements a rolling-window approach where:
    1. Parameters are estimated on a lookback window
    2. Allocation is computed and held for rebalance_freq periods
    3. Process repeats, walking forward through time
    
    Parameters
    ----------
    lookback : int
        Number of periods for parameter estimation (e.g., 252 for 1 year daily)
    rebalance_freq : int
        Number of periods between rebalances (e.g., 21 for monthly)
    cost_bps : float
        Transaction cost in basis points (round-trip)
    risk_free_rate : float
        Risk-free rate (same frequency as returns)
    """
    
    def __init__(
        self,
        lookback: int = 252,
        rebalance_freq: int = 21,
        cost_bps: float = 10.0,
        risk_free_rate: float = 0.0,
    ):
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.cost_bps = cost_bps
        self.risk_free_rate = risk_free_rate
    
    def run(
        self,
        returns: pd.DataFrame,
        strategy_fn: Callable[[np.ndarray], np.ndarray],
        initial_wealth: float = 1.0,
    ) -> BacktestResult:
        """
        Run walk-forward backtest.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return series, shape (T, d) where T is time and d is assets.
            Index should be datetime-like.
        strategy_fn : Callable
            Function that takes historical returns (n, d) and returns
            target weights (d,). For single asset, d=1.
        initial_wealth : float
            Starting wealth
            
        Returns
        -------
        BacktestResult
            Complete backtest results
        """
        # Ensure DataFrame format
        if isinstance(returns, pd.Series):
            returns = returns.to_frame()
        
        T, d = returns.shape
        
        if T < self.lookback + self.rebalance_freq:
            raise ValueError(
                f"Insufficient data: need {self.lookback + self.rebalance_freq} "
                f"periods, got {T}"
            )
        
        # Initialize tracking
        wealth = initial_wealth
        wealth_history = []
        weights = np.zeros(d)
        weights_history = []
        returns_history = []
        turnover_total = 0.0
        costs_total = 0.0
        num_rebalances = 0
        
        # Walk forward
        t = self.lookback
        while t < T:
            # === Rebalance Point ===
            # Get historical window for estimation
            hist_returns = returns.iloc[t - self.lookback:t].values
            
            # Compute target weights
            try:
                new_weights = strategy_fn(hist_returns)
                new_weights = np.atleast_1d(new_weights).flatten()
                
                # Validate weights
                if len(new_weights) != d:
                    raise ValueError(f"Strategy returned {len(new_weights)} weights, expected {d}")
                if np.any(np.isnan(new_weights)):
                    warnings.warn(f"NaN weights at t={t}, using previous weights")
                    new_weights = weights.copy()
                    
            except Exception as e:
                warnings.warn(f"Strategy error at t={t}: {e}. Using previous weights.")
                new_weights = weights.copy()
            
            # Transaction costs
            turnover = np.sum(np.abs(new_weights - weights))
            cost = turnover * (self.cost_bps / 10000)
            wealth *= (1 - cost)
            
            turnover_total += turnover
            costs_total += cost * wealth
            num_rebalances += 1
            
            # Update weights
            weights = new_weights.copy()
            
            # === Simulate until next rebalance ===
            end_t = min(t + self.rebalance_freq, T)
            
            for tau in range(t, end_t):
                # Record state
                weights_history.append({
                    'date': returns.index[tau],
                    **{f'w_{i}': weights[i] for i in range(d)}
                })
                
                # Compute portfolio return
                period_returns = returns.iloc[tau].values
                
                # Portfolio return: w'r + (1 - sum(w)) * r_f
                portfolio_return = (
                    np.dot(weights, period_returns) +
                    (1 - np.sum(weights)) * self.risk_free_rate
                )
                
                # Update wealth
                wealth *= (1 + portfolio_return)
                
                # Record
                wealth_history.append({
                    'date': returns.index[tau],
                    'wealth': wealth
                })
                returns_history.append({
                    'date': returns.index[tau],
                    'return': portfolio_return
                })
            
            # Move forward
            t = end_t
        
        # Convert to DataFrames/Series
        wealth_df = pd.DataFrame(wealth_history).set_index('date')
        returns_df = pd.DataFrame(returns_history).set_index('date')
        weights_df = pd.DataFrame(weights_history).set_index('date')
        
        wealth_series = wealth_df['wealth']
        returns_series = returns_df['return']
        
        # Compute performance metrics
        metrics = self._compute_metrics(wealth_series, returns_series)
        
        return BacktestResult(
            wealth_series=wealth_series,
            returns_series=returns_series,
            weights_history=weights_df,
            total_return=metrics['total_return'],
            annualized_return=metrics['annualized_return'],
            annualized_volatility=metrics['annualized_volatility'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            calmar_ratio=metrics['calmar_ratio'],
            total_turnover=turnover_total,
            total_costs=costs_total,
            num_rebalances=num_rebalances,
            metadata={
                'lookback': self.lookback,
                'rebalance_freq': self.rebalance_freq,
                'cost_bps': self.cost_bps,
                'periods': len(wealth_series),
            }
        )
    
    def _compute_metrics(
        self,
        wealth: pd.Series,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> Dict[str, float]:
        """Compute performance metrics."""
        
        # Total return
        total_return = wealth.iloc[-1] / wealth.iloc[0] - 1
        
        # Annualized return
        n_years = len(wealth) / periods_per_year
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Volatility
        annualized_volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        excess_return = annualized_return - self.risk_free_rate * periods_per_year
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Drawdown
        running_max = wealth.expanding().max()
        drawdown = (wealth - running_max) / running_max
        max_drawdown = -drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
        }


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    strategy_fn: Callable[[np.ndarray], np.ndarray],
    lookback: int = 252,
    rebalance_freq: int = 21,
    cost_bps: float = 10.0,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """
    Convenience function to run walk-forward backtest from prices.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price series (T, d)
    strategy_fn : Callable
        Strategy function: hist_returns → weights
    lookback : int
        Estimation window
    rebalance_freq : int
        Rebalancing frequency
    cost_bps : float
        Transaction costs in basis points
    risk_free_rate : float
        Risk-free rate per period
        
    Returns
    -------
    BacktestResult
        Backtest results
    """
    # Convert prices to returns
    returns = prices.pct_change().dropna()
    
    # Run backtest
    engine = WalkForwardBacktest(
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        cost_bps=cost_bps,
        risk_free_rate=risk_free_rate,
    )
    
    return engine.run(returns, strategy_fn)


def compare_strategies(
    returns: pd.DataFrame,
    strategies: Dict[str, Callable[[np.ndarray], np.ndarray]],
    lookback: int = 252,
    rebalance_freq: int = 21,
    cost_bps: float = 10.0,
    risk_free_rate: float = 0.0,
) -> Dict[str, BacktestResult]:
    """
    Compare multiple strategies on the same data.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return series
    strategies : Dict[str, Callable]
        Dictionary of strategy_name → strategy_function
    lookback, rebalance_freq, cost_bps, risk_free_rate : 
        Backtest parameters
        
    Returns
    -------
    Dict[str, BacktestResult]
        Results for each strategy
    """
    engine = WalkForwardBacktest(
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        cost_bps=cost_bps,
        risk_free_rate=risk_free_rate,
    )
    
    results = {}
    for name, fn in strategies.items():
        try:
            results[name] = engine.run(returns, fn)
        except Exception as e:
            warnings.warn(f"Strategy '{name}' failed: {e}")
            results[name] = None
    
    return results


def print_comparison_table(results: Dict[str, BacktestResult]) -> None:
    """Print a formatted comparison table."""
    
    headers = ['Strategy', 'Return', 'Vol', 'Sharpe', 'MaxDD', 'Calmar', 'Turnover']
    
    print("\n" + "=" * 80)
    print(f"{'Strategy Comparison':^80}")
    print("=" * 80)
    print(f"{headers[0]:<15} {headers[1]:>10} {headers[2]:>10} {headers[3]:>10} "
          f"{headers[4]:>10} {headers[5]:>10} {headers[6]:>10}")
    print("-" * 80)
    
    for name, result in results.items():
        if result is None:
            print(f"{name:<15} {'FAILED':>10}")
            continue
            
        print(
            f"{name:<15} "
            f"{result.annualized_return:>10.2%} "
            f"{result.annualized_volatility:>10.2%} "
            f"{result.sharpe_ratio:>10.3f} "
            f"{result.max_drawdown:>10.2%} "
            f"{result.calmar_ratio:>10.2f} "
            f"{result.total_turnover:>10.1f}x"
        )
    
    print("=" * 80)

