"""
Monte Carlo Simulation Engine for Kelly Strategy Evaluation.

Provides:
- Wealth path simulation
- Growth metric computation
- Strategy comparison framework
- Sensitivity analysis

Author: Agna Chan
Date: December 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
import warnings


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation results."""
    strategy_name: str
    fractions: np.ndarray
    terminal_wealth: np.ndarray
    log_wealth: np.ndarray
    max_drawdowns: np.ndarray
    
    @property
    def mean_fraction(self) -> float:
        return np.mean(self.fractions)
    
    @property
    def mean_terminal_wealth(self) -> float:
        return np.mean(self.terminal_wealth)
    
    @property
    def median_terminal_wealth(self) -> float:
        return np.median(self.terminal_wealth)
    
    @property
    def mean_log_wealth(self) -> float:
        return np.mean(self.log_wealth)
    
    @property
    def median_log_wealth(self) -> float:
        return np.median(self.log_wealth)
    
    @property
    def prob_profit(self) -> float:
        return np.mean(self.terminal_wealth > 1.0)
    
    @property
    def mean_max_drawdown(self) -> float:
        return np.mean(self.max_drawdowns)
    
    def summary(self) -> dict:
        """Return summary statistics."""
        return {
            'mean_fraction': self.mean_fraction,
            'std_fraction': np.std(self.fractions),
            'mean_terminal_wealth': self.mean_terminal_wealth,
            'median_terminal_wealth': self.median_terminal_wealth,
            'std_terminal_wealth': np.std(self.terminal_wealth),
            'mean_log_wealth': self.mean_log_wealth,
            'median_log_wealth': self.median_log_wealth,
            'prob_profit': self.prob_profit,
            'prob_double': np.mean(self.terminal_wealth > 2.0),
            'prob_ruin': np.mean(self.terminal_wealth < 0.1),
            'mean_max_drawdown': self.mean_max_drawdown,
            'median_max_drawdown': np.median(self.max_drawdowns),
        }


def simulate_wealth_paths(
    returns: np.ndarray,
    fraction: float,
    risk_free: float = 0.0,
    initial_wealth: float = 1.0,
) -> np.ndarray:
    """
    Simulate wealth paths given returns and allocation fraction.
    
    W_{t+1} = W_t × [1 + r + f(R_t - r)]
    
    where R_t = exp(X_t) - 1 for log-returns X_t.
    
    Parameters
    ----------
    returns : np.ndarray
        Log-returns, shape (n_paths, n_periods) or (n_periods,)
    fraction : float
        Allocation to risky asset
    risk_free : float
        Risk-free rate per period
    initial_wealth : float
        Starting wealth
        
    Returns
    -------
    np.ndarray
        Wealth paths, shape (n_paths, n_periods+1) or (n_periods+1,)
    """
    returns = np.atleast_2d(returns)
    n_paths, n_periods = returns.shape
    
    # Convert log-returns to simple returns
    simple_returns = np.exp(returns) - 1
    
    # Portfolio returns
    portfolio_returns = risk_free + fraction * (simple_returns - risk_free)
    
    # Wealth accumulation
    wealth = np.zeros((n_paths, n_periods + 1))
    wealth[:, 0] = initial_wealth
    
    for t in range(n_periods):
        wealth[:, t+1] = wealth[:, t] * (1 + portfolio_returns[:, t])
    
    return wealth if n_paths > 1 else wealth.flatten()


def compute_growth_metrics(wealth_paths: np.ndarray) -> dict:
    """
    Compute growth-related performance metrics.
    
    Parameters
    ----------
    wealth_paths : np.ndarray
        Wealth paths, shape (n_paths, T+1)
        
    Returns
    -------
    dict
        Dictionary of performance metrics
    """
    wealth_paths = np.atleast_2d(wealth_paths)
    n_paths, T_plus_1 = wealth_paths.shape
    T = T_plus_1 - 1
    
    # Terminal wealth
    terminal_wealth = wealth_paths[:, -1]
    initial_wealth = wealth_paths[:, 0]
    
    # Log-wealth
    log_terminal = np.log(np.maximum(terminal_wealth, 1e-10))
    
    # Realized growth rate (per period)
    realized_growth = log_terminal / T
    
    # Drawdowns
    running_max = np.maximum.accumulate(wealth_paths, axis=1)
    drawdowns = (running_max - wealth_paths) / np.maximum(running_max, 1e-10)
    max_drawdown = np.max(drawdowns, axis=1)
    
    return {
        'mean_terminal_wealth': np.mean(terminal_wealth),
        'median_terminal_wealth': np.median(terminal_wealth),
        'std_terminal_wealth': np.std(terminal_wealth),
        'mean_log_wealth': np.mean(log_terminal),
        'median_log_wealth': np.median(log_terminal),
        'mean_growth_rate': np.mean(realized_growth),
        'median_growth_rate': np.median(realized_growth),
        'prob_profit': np.mean(terminal_wealth > initial_wealth),
        'prob_double': np.mean(terminal_wealth > 2 * initial_wealth),
        'prob_ruin': np.mean(terminal_wealth < 0.01 * initial_wealth),
        'mean_max_drawdown': np.mean(max_drawdown),
        'median_max_drawdown': np.median(max_drawdown),
        'max_max_drawdown': np.max(max_drawdown),
        'q05_terminal': np.percentile(terminal_wealth, 5),
        'q95_terminal': np.percentile(terminal_wealth, 95),
    }


def monte_carlo_kelly_comparison(
    true_mu: float,
    true_sigma: float,
    sample_size: int,
    horizon: int,
    n_simulations: int = 1000,
    risk_free: float = 0.0,
    alpha: float = 0.1,
    seed: Optional[int] = None,
    return_simulator: Optional[Callable] = None,
) -> Dict[str, SimulationResult]:
    """
    Compare Kelly strategies across Monte Carlo simulations.
    
    Strategies compared:
    1. Oracle Kelly (knows true parameters)
    2. Plug-in Kelly (estimated parameters)
    3. Half-Kelly (50% of plug-in)
    4. DRK (distributionally robust)
    5. ACK (adaptive conformal Kelly)
    
    Parameters
    ----------
    true_mu : float
        True mean return
    true_sigma : float
        True volatility
    sample_size : int
        Historical data length for estimation
    horizon : int
        Investment horizon
    n_simulations : int
        Number of MC runs
    risk_free : float
        Risk-free rate
    alpha : float
        Confidence level for DRK
    seed : int, optional
        Random seed
    return_simulator : Callable, optional
        Custom return generator (default: GBM)
        
    Returns
    -------
    Dict[str, SimulationResult]
        Results for each strategy
    """
    # Import core functions
    import sys
    sys.path.insert(0, 'src')
    from kelly_robust.core.kelly import (
        kelly_single_asset,
        drk_single_asset_closed_form,
        adaptive_conformal_kelly,
    )
    from .return_models import simulate_gbm_returns
    
    if seed is not None:
        np.random.seed(seed)
    
    true_sigma2 = true_sigma ** 2
    
    # Default simulator: GBM
    if return_simulator is None:
        return_simulator = lambda n, seed=None: simulate_gbm_returns(
            true_mu, true_sigma, n, n_paths=1, seed=seed
        )
    
    # Oracle Kelly fraction
    f_oracle = kelly_single_asset(true_mu, true_sigma2, risk_free)
    f_oracle = np.clip(f_oracle, 0, 1)
    
    # Initialize result containers
    results = {
        'oracle': {'fractions': [], 'terminal': [], 'log_wealth': [], 'mdd': []},
        'plugin': {'fractions': [], 'terminal': [], 'log_wealth': [], 'mdd': []},
        'half': {'fractions': [], 'terminal': [], 'log_wealth': [], 'mdd': []},
        'drk': {'fractions': [], 'terminal': [], 'log_wealth': [], 'mdd': []},
        'ack': {'fractions': [], 'terminal': [], 'log_wealth': [], 'mdd': []},
    }
    
    for sim in range(n_simulations):
        # Generate historical data
        hist_returns = return_simulator(sample_size)
        
        # Generate forward returns
        fwd_returns = return_simulator(horizon)
        
        # === Estimate parameters ===
        mu_hat = np.mean(hist_returns)
        sigma2_hat = np.var(hist_returns, ddof=1)
        
        # === Plug-in Kelly ===
        try:
            f_plugin = kelly_single_asset(mu_hat, sigma2_hat, risk_free)
            f_plugin = np.clip(f_plugin, 0, 1)
        except:
            f_plugin = 0
        
        # === Half-Kelly ===
        f_half = 0.5 * f_plugin
        
        # === DRK (standard error based) ===
        se = np.sqrt(max(sigma2_hat, 1e-10) / sample_size)
        epsilon = 1.645 * se  # 90% confidence
        try:
            f_drk = drk_single_asset_closed_form(
                mu_hat, sigma2_hat, epsilon, risk_free,
                min_fraction=0, max_fraction=1
            )
        except:
            f_drk = 0
        
        # === ACK (conformal) ===
        try:
            ack_result = adaptive_conformal_kelly(
                hist_returns, risk_free, alpha=alpha,
                split_ratio=0.5, min_fraction=0, max_fraction=1
            )
            f_ack = ack_result.fraction
        except:
            f_ack = 0
        
        # === Simulate wealth for each strategy ===
        for name, frac in [
            ('oracle', f_oracle),
            ('plugin', f_plugin),
            ('half', f_half),
            ('drk', f_drk),
            ('ack', f_ack),
        ]:
            wealth = simulate_wealth_paths(fwd_returns, frac, risk_free)
            
            # Compute drawdown
            running_max = np.maximum.accumulate(wealth)
            dd = (running_max - wealth) / np.maximum(running_max, 1e-10)
            mdd = np.max(dd)
            
            results[name]['fractions'].append(frac)
            results[name]['terminal'].append(wealth[-1])
            results[name]['log_wealth'].append(np.log(max(wealth[-1], 1e-10)))
            results[name]['mdd'].append(mdd)
    
    # Convert to SimulationResult objects
    simulation_results = {}
    for name, data in results.items():
        simulation_results[name] = SimulationResult(
            strategy_name=name,
            fractions=np.array(data['fractions']),
            terminal_wealth=np.array(data['terminal']),
            log_wealth=np.array(data['log_wealth']),
            max_drawdowns=np.array(data['mdd']),
        )
    
    return simulation_results


def parameter_sensitivity_analysis(
    base_mu: float = 0.0005,
    base_sigma: float = 0.02,
    sample_sizes: List[int] = [50, 100, 252, 504],
    horizons: List[int] = [252],
    n_simulations: int = 500,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Analyze sensitivity to sample size and other parameters.
    
    Parameters
    ----------
    base_mu : float
        Base mean return
    base_sigma : float
        Base volatility
    sample_sizes : List[int]
        Sample sizes to test
    horizons : List[int]
        Investment horizons to test
    n_simulations : int
        MC simulations per configuration
    seed : int, optional
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Sensitivity analysis results
    """
    if seed is not None:
        np.random.seed(seed)
    
    results = []
    
    for n in sample_sizes:
        for h in horizons:
            mc_results = monte_carlo_kelly_comparison(
                true_mu=base_mu,
                true_sigma=base_sigma,
                sample_size=n,
                horizon=h,
                n_simulations=n_simulations,
            )
            
            for strategy, res in mc_results.items():
                summary = res.summary()
                summary['strategy'] = strategy
                summary['sample_size'] = n
                summary['horizon'] = h
                results.append(summary)
    
    return pd.DataFrame(results)


def print_comparison_summary(results: Dict[str, SimulationResult]) -> None:
    """Print formatted comparison table."""
    
    print("\n" + "=" * 85)
    print(f"{'Kelly Strategy Comparison':^85}")
    print("=" * 85)
    
    headers = ['Strategy', 'Mean f', 'Std f', 'Mean W_T', 'Med W_T', 
               'Mean log(W)', 'P(profit)', 'Mean MDD']
    
    print(f"{headers[0]:<10} {headers[1]:>10} {headers[2]:>10} {headers[3]:>12} "
          f"{headers[4]:>10} {headers[5]:>12} {headers[6]:>10} {headers[7]:>10}")
    print("-" * 85)
    
    for name in ['oracle', 'plugin', 'half', 'drk', 'ack']:
        if name not in results:
            continue
        r = results[name]
        s = r.summary()
        print(
            f"{name:<10} "
            f"{s['mean_fraction']:>10.4f} "
            f"{s['std_fraction']:>10.4f} "
            f"{s['mean_terminal_wealth']:>12.4f} "
            f"{s['median_terminal_wealth']:>10.4f} "
            f"{s['mean_log_wealth']:>12.4f} "
            f"{s['prob_profit']:>10.2%} "
            f"{s['mean_max_drawdown']:>10.2%}"
        )
    
    print("=" * 85)


def regret_analysis(
    results: Dict[str, SimulationResult],
    oracle_name: str = 'oracle',
) -> pd.DataFrame:
    """
    Compute regret relative to oracle strategy.
    
    Parameters
    ----------
    results : Dict[str, SimulationResult]
        MC simulation results
    oracle_name : str
        Name of the oracle strategy
        
    Returns
    -------
    pd.DataFrame
        Regret statistics
    """
    oracle = results[oracle_name]
    
    regret_data = []
    for name, res in results.items():
        if name == oracle_name:
            continue
        
        # Log-wealth regret
        log_regret = oracle.log_wealth - res.log_wealth
        
        regret_data.append({
            'strategy': name,
            'mean_log_regret': np.mean(log_regret),
            'median_log_regret': np.median(log_regret),
            'std_log_regret': np.std(log_regret),
            'prob_outperform': np.mean(log_regret < 0),
            'max_regret': np.max(log_regret),
            'q95_regret': np.percentile(log_regret, 95),
        })
    
    return pd.DataFrame(regret_data)

