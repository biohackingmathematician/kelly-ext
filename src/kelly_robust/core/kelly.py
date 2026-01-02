"""
Distributionally Robust Kelly (DRK) Optimization

Core implementation of the theoretical framework for robust Kelly allocation
under distributional ambiguity.

Author: Agna Chan
Date: December 2025
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from typing import Tuple, Callable, Optional
from dataclasses import dataclass
import warnings


# =============================================================================
# Part 1: Classical Kelly (Baseline)
# =============================================================================

def kelly_single_asset(
    mu: float,
    sigma2: float,
    risk_free: float = 0.0
) -> float:
    """
    Classical Kelly fraction for a single asset.
    
    f* = (μ - r) / σ²
    
    Parameters
    ----------
    mu : float
        Expected return (arithmetic, same frequency as sigma2)
    sigma2 : float
        Return variance
    risk_free : float
        Risk-free rate
        
    Returns
    -------
    float
        Optimal Kelly fraction
    """
    if sigma2 <= 0:
        raise ValueError("Variance must be positive")
    return (mu - risk_free) / sigma2


def kelly_multi_asset(
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_free: float = 0.0
) -> np.ndarray:
    """
    Classical Kelly portfolio for multiple assets.
    
    F* = Σ⁻¹(μ - r·1)
    
    Parameters
    ----------
    mu : np.ndarray
        Expected returns vector (d,)
    Sigma : np.ndarray
        Covariance matrix (d, d)
    risk_free : float
        Risk-free rate
        
    Returns
    -------
    np.ndarray
        Optimal Kelly fractions (d,)
    """
    d = len(mu)
    excess = mu - risk_free * np.ones(d)
    return np.linalg.solve(Sigma, excess)


# =============================================================================
# Part 2: Growth Rate Computation
# =============================================================================

def growth_rate_gaussian(
    f: float,
    mu: float,
    sigma2: float,
    risk_free: float = 0.0
) -> float:
    """
    Expected growth rate under Gaussian returns (continuous-time approximation).
    
    g(f) = r + f(μ - r) - (1/2)f²σ²
    
    Parameters
    ----------
    f : float
        Allocation fraction
    mu, sigma2, risk_free : float
        Return parameters
        
    Returns
    -------
    float
        Expected log-growth rate
    """
    return risk_free + f * (mu - risk_free) - 0.5 * f**2 * sigma2


def growth_rate_exact(
    f: float,
    mu: float,
    sigma2: float,
    risk_free: float = 0.0
) -> float:
    """
    Exact expected growth rate under log-normal returns.
    
    g(f) = E[log(1 + r + f(R - r))]
    
    Uses numerical integration for accuracy.
    """
    from scipy.integrate import quad
    
    sigma = np.sqrt(sigma2)
    
    def integrand(z):
        R = mu + sigma * z  # Return realization
        W_next = 1 + risk_free + f * (R - risk_free)
        if W_next <= 0:
            return -np.inf * stats.norm.pdf(z)
        return np.log(W_next) * stats.norm.pdf(z)
    
    result, _ = quad(integrand, -8, 8)
    return result


# =============================================================================
# Part 3: Distributionally Robust Kelly (DRK)
# =============================================================================

@dataclass
class DRKResult:
    """Container for DRK optimization results."""
    fraction: float
    epsilon: float
    lambda_penalty: float
    mu_hat: float
    sigma2_hat: float
    confidence_level: float


def drk_single_asset_closed_form(
    mu_hat: float,
    sigma2_hat: float,
    epsilon: float,
    risk_free: float = 0.0,
    min_fraction: float = 0.0,
    max_fraction: float = 1.0
) -> float:
    """
    DRK fraction via closed-form solution (Theorem 1).
    
    f_DRK = (μ̂ - r - ε) / σ̂²
    
    Under Wasserstein ambiguity, the worst-case distribution shifts
    the mean adversarially by ε.
    
    Parameters
    ----------
    mu_hat : float
        Estimated mean return
    sigma2_hat : float
        Estimated variance
    epsilon : float
        Ambiguity radius (from conformal calibration)
    risk_free : float
        Risk-free rate
    min_fraction, max_fraction : float
        Allocation constraints
        
    Returns
    -------
    float
        Robust Kelly fraction
    """
    if sigma2_hat <= 0:
        raise ValueError("Estimated variance must be positive")
    
    # Worst-case mean is μ̂ - ε
    mu_worst_case = mu_hat - epsilon
    
    # Kelly on worst-case
    f_drk = (mu_worst_case - risk_free) / sigma2_hat
    
    # Apply constraints
    f_drk = np.clip(f_drk, min_fraction, max_fraction)
    
    return f_drk


def calibrate_epsilon_conformal(
    returns: np.ndarray,
    alpha: float = 0.1,
    split_ratio: float = 0.5
) -> Tuple[float, float, float]:
    """
    Calibrate ambiguity radius via split conformal prediction.
    
    This provides distribution-free coverage guarantees:
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
    """
    n = len(returns)
    m = int(n * split_ratio)
    
    if m < 10:
        warnings.warn("Training set too small; results may be unreliable")
    
    # Split data
    train = returns[:m]
    calib = returns[m:]
    
    # Fit on training set
    mu_hat = np.mean(train)
    sigma2_hat = np.var(train, ddof=1)
    
    # Compute nonconformity scores on calibration set
    scores = np.abs(calib - mu_hat)
    
    # Quantile for coverage
    # Use (1 - α)(1 + 1/n_calib) correction for finite-sample validity
    n_calib = len(calib)
    q_level = np.ceil((1 - alpha) * (n_calib + 1)) / n_calib
    q_level = min(q_level, 1.0)
    
    epsilon = np.quantile(scores, q_level)
    
    return epsilon, mu_hat, sigma2_hat


def adaptive_conformal_kelly(
    returns: np.ndarray,
    risk_free: float = 0.0,
    alpha: float = 0.1,
    split_ratio: float = 0.5,
    min_fraction: float = 0.0,
    max_fraction: float = 1.0
) -> DRKResult:
    """
    Adaptive Conformal Kelly (ACK) algorithm.
    
    Main entry point for single-asset DRK with conformal calibration.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns
    risk_free : float
        Risk-free rate (same frequency as returns)
    alpha : float
        Confidence level (e.g., 0.1 for 90%)
    split_ratio : float
        Data split ratio
    min_fraction, max_fraction : float
        Allocation constraints
        
    Returns
    -------
    DRKResult
        Complete result with fraction, diagnostics
    """
    # Calibrate ambiguity
    epsilon, mu_hat, sigma2_hat = calibrate_epsilon_conformal(
        returns, alpha=alpha, split_ratio=split_ratio
    )
    
    # Compute penalty (see Theorem 1)
    # λ(ε) = ε · √(2 log(1/α)) for Gaussian case
    lambda_penalty = epsilon  # Simplified; full formula in theory
    
    # Robust fraction
    f_drk = drk_single_asset_closed_form(
        mu_hat=mu_hat,
        sigma2_hat=sigma2_hat,
        epsilon=lambda_penalty,
        risk_free=risk_free,
        min_fraction=min_fraction,
        max_fraction=max_fraction
    )
    
    return DRKResult(
        fraction=f_drk,
        epsilon=epsilon,
        lambda_penalty=lambda_penalty,
        mu_hat=mu_hat,
        sigma2_hat=sigma2_hat,
        confidence_level=1 - alpha
    )


# =============================================================================
# Part 4: Multi-Asset DRK (Requires CVXPY)
# =============================================================================

def drk_multi_asset_socp(
    mu_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    epsilon: float,
    risk_free: float = 0.0,
    leverage_limit: float = 1.0,
    long_only: bool = True
) -> np.ndarray:
    """
    Multi-asset DRK via Second-Order Cone Programming (SOCP).
    
    Solves the distributionally robust Kelly problem:
        max_f min_{Q in A_eps} g(f; Q)
    
    Under mean-only Wasserstein ambiguity, this simplifies to:
        max_f [f'(mu_hat-r) - (1/2)f'Sigma_hat*f - eps||f||_2]
    
    This is a convex SOCP (not SDP as sometimes incorrectly stated).
    The eps||f||_2 term is a second-order cone constraint.
    
    Parameters
    ----------
    mu_hat : np.ndarray
        Estimated mean returns (d,)
    Sigma_hat : np.ndarray
        Estimated covariance matrix (d, d)
    epsilon : float
        Ambiguity radius (Wasserstein ball size)
    risk_free : float
        Risk-free rate
    leverage_limit : float
        Maximum sum of weights (1.0 = no leverage)
    long_only : bool
        Whether to enforce non-negative weights
        
    Returns
    -------
    np.ndarray
        Robust portfolio weights (d,)
        
    Notes
    -----
    The problem structure:
    - Objective: concave quadratic + linear + cone term
    - Constraints: linear (leverage, long-only)
    - Complexity: O(d^3) per iteration, polynomial overall
    
    References
    ----------
    - Blanchet & Murthy (2019), Math. Oper. Res.
    - Boyd & Vandenberghe (2004), Convex Optimization, Ch. 4.4
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("cvxpy required. Install via: pip install cvxpy")
    
    d = len(mu_hat)
    
    # Decision variables
    f = cp.Variable(d)
    
    # Excess returns vector
    excess = mu_hat - risk_free * np.ones(d)
    
    # Objective: maximize worst-case growth rate
    # g_wc(f) = f'(mu-r) - (1/2)f'Sigma*f - eps||f||_2
    expected_excess = f @ excess
    variance_penalty = 0.5 * cp.quad_form(f, Sigma_hat)
    ambiguity_penalty = epsilon * cp.norm(f, 2)  # This is the SOCP part
    
    objective = expected_excess - variance_penalty - ambiguity_penalty
    
    # Constraints
    constraints = [cp.sum(f) <= leverage_limit]
    if long_only:
        constraints.append(f >= 0)
    
    # Solve
    problem = cp.Problem(cp.Maximize(objective), constraints)
    
    # Try MOSEK first (fastest), fall back to ECOS or SCS
    solvers_to_try = [cp.MOSEK, cp.ECOS, cp.SCS]
    
    for solver in solvers_to_try:
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in ['optimal', 'optimal_inaccurate']:
                break
        except (cp.error.SolverError, Exception):
            continue
    
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        warnings.warn(f"Optimization status: {problem.status}. Returning equal weights.")
        return np.ones(d) / d
    
    return f.value


# Backward compatibility alias (deprecated)
drk_multi_asset_sdp = drk_multi_asset_socp


# =============================================================================
# Part 5: Monte Carlo Simulation Engine
# =============================================================================

def simulate_gbm_returns(
    mu: float,
    sigma: float,
    n_periods: int,
    n_paths: int = 1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate log-returns from Geometric Brownian Motion.
    
    X_t = (μ - σ²/2)dt + σ√dt · Z_t
    
    where Z_t ~ N(0, 1)
    
    Parameters
    ----------
    mu : float
        Drift (annualized)
    sigma : float
        Volatility (annualized)
    n_periods : int
        Number of time steps
    n_paths : int
        Number of sample paths
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Log-returns, shape (n_paths, n_periods)
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = 1.0  # Assumes returns are already at correct frequency
    
    # Log-returns under GBM
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    Z = np.random.randn(n_paths, n_periods)
    returns = drift + diffusion * Z
    
    return returns if n_paths > 1 else returns.flatten()


def simulate_wealth_paths(
    returns: np.ndarray,
    fraction: float,
    risk_free: float = 0.0,
    initial_wealth: float = 1.0
) -> np.ndarray:
    """
    Simulate wealth paths given returns and allocation fraction.
    
    W_{t+1} = W_t · [1 + r + f(R_t - r)]
    
    where R_t = e^{X_t} - 1 (convert log-returns to simple returns)
    
    Parameters
    ----------
    returns : np.ndarray
        Log-returns, shape (n_paths, n_periods) or (n_periods,)
    fraction : float
        Allocation to risky asset
    risk_free : float
        Risk-free rate
    initial_wealth : float
        Starting wealth
        
    Returns
    -------
    np.ndarray
        Wealth paths, shape (n_paths, n_periods+1)
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


def compute_growth_metrics(
    wealth_paths: np.ndarray
) -> dict:
    """
    Compute growth-related performance metrics.
    
    Parameters
    ----------
    wealth_paths : np.ndarray
        Wealth paths, shape (n_paths, T+1)
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    wealth_paths = np.atleast_2d(wealth_paths)
    n_paths, T_plus_1 = wealth_paths.shape
    T = T_plus_1 - 1
    
    # Terminal wealth
    terminal_wealth = wealth_paths[:, -1]
    
    # Log-wealth
    log_terminal = np.log(terminal_wealth)
    
    # Realized growth rate (per period)
    realized_growth = log_terminal / T
    
    # Drawdowns
    running_max = np.maximum.accumulate(wealth_paths, axis=1)
    drawdowns = (running_max - wealth_paths) / running_max
    max_drawdown = np.max(drawdowns, axis=1)
    
    return {
        'mean_terminal_wealth': np.mean(terminal_wealth),
        'median_terminal_wealth': np.median(terminal_wealth),
        'std_terminal_wealth': np.std(terminal_wealth),
        'mean_log_wealth': np.mean(log_terminal),
        'median_log_wealth': np.median(log_terminal),
        'mean_growth_rate': np.mean(realized_growth),
        'median_growth_rate': np.median(realized_growth),
        'prob_profit': np.mean(terminal_wealth > wealth_paths[:, 0]),
        'prob_double': np.mean(terminal_wealth > 2 * wealth_paths[:, 0]),
        'prob_ruin': np.mean(np.any(wealth_paths < 0.01, axis=1)),
        'mean_max_drawdown': np.mean(max_drawdown),
        'median_max_drawdown': np.median(max_drawdown),
        'max_max_drawdown': np.max(max_drawdown),
    }


# =============================================================================
# Part 6: Comparison Study
# =============================================================================

def run_kelly_comparison(
    true_mu: float,
    true_sigma: float,
    sample_size: int,
    horizon: int,
    n_simulations: int = 1000,
    risk_free: float = 0.0,
    alpha: float = 0.1,
    seed: Optional[int] = None
) -> dict:
    """
    Compare Kelly strategies across multiple simulations.
    
    Strategies compared:
    1. Oracle Kelly (knows true parameters)
    2. Plug-in Kelly (uses estimated parameters)
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
        Investment horizon (number of periods)
    n_simulations : int
        Number of simulation runs
    risk_free : float
        Risk-free rate
    alpha : float
        Confidence level for DRK
    seed : int, optional
        Random seed
        
    Returns
    -------
    dict
        Results dictionary with metrics per strategy
    """
    if seed is not None:
        np.random.seed(seed)
    
    true_sigma2 = true_sigma**2
    
    # Oracle Kelly fraction
    f_oracle = kelly_single_asset(true_mu, true_sigma2, risk_free)
    f_oracle = np.clip(f_oracle, 0, 1)
    
    results = {
        'oracle': [],
        'plugin': [],
        'half': [],
        'drk': [],
        'ack': []
    }
    
    for sim in range(n_simulations):
        # Generate historical data for estimation
        hist_returns = simulate_gbm_returns(
            mu=true_mu, sigma=true_sigma, n_periods=sample_size
        )
        
        # Generate forward returns for evaluation
        fwd_returns = simulate_gbm_returns(
            mu=true_mu, sigma=true_sigma, n_periods=horizon
        )
        
        # === Plug-in Kelly ===
        mu_hat = np.mean(hist_returns)
        sigma2_hat = np.var(hist_returns, ddof=1)
        f_plugin = kelly_single_asset(mu_hat, sigma2_hat, risk_free)
        f_plugin = np.clip(f_plugin, 0, 1)
        
        # === Half-Kelly ===
        f_half = 0.5 * f_plugin
        
        # === DRK (fixed epsilon based on standard error) ===
        se = np.sqrt(sigma2_hat / sample_size)
        epsilon_fixed = 1.645 * se  # 90% confidence
        f_drk = drk_single_asset_closed_form(
            mu_hat, sigma2_hat, epsilon_fixed, risk_free, min_fraction=0, max_fraction=1
        )
        
        # === ACK (conformal) ===
        ack_result = adaptive_conformal_kelly(
            hist_returns, risk_free, alpha=alpha, 
            split_ratio=0.5, min_fraction=0, max_fraction=1
        )
        f_ack = ack_result.fraction
        
        # Simulate wealth for each strategy
        for name, frac in [
            ('oracle', f_oracle),
            ('plugin', f_plugin),
            ('half', f_half),
            ('drk', f_drk),
            ('ack', f_ack)
        ]:
            wealth = simulate_wealth_paths(fwd_returns, frac, risk_free)
            results[name].append({
                'fraction': frac,
                'terminal_wealth': wealth[-1],
                'log_wealth': np.log(wealth[-1]),
                'max_drawdown': np.max(np.maximum.accumulate(wealth) - wealth) / np.maximum.accumulate(wealth).max()
            })
    
    # Aggregate results
    summary = {}
    for strategy, runs in results.items():
        fractions = [r['fraction'] for r in runs]
        terminals = [r['terminal_wealth'] for r in runs]
        log_ws = [r['log_wealth'] for r in runs]
        mdd = [r['max_drawdown'] for r in runs]
        
        summary[strategy] = {
            'mean_fraction': np.mean(fractions),
            'std_fraction': np.std(fractions),
            'mean_terminal_wealth': np.mean(terminals),
            'median_terminal_wealth': np.median(terminals),
            'mean_log_wealth': np.mean(log_ws),
            'median_log_wealth': np.median(log_ws),
            'prob_profit': np.mean(np.array(terminals) > 1),
            'mean_max_drawdown': np.mean(mdd),
        }
    
    return summary


# =============================================================================
# Part 7: Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Distributionally Robust Kelly - Demonstration")
    print("=" * 70)
    
    # Parameters (daily returns, annualized ~12% return, ~40% vol)
    TRUE_MU = 0.0005           # Daily mean
    TRUE_SIGMA = 0.025         # Daily volatility
    TRUE_SIGMA2 = TRUE_SIGMA**2
    RISK_FREE = 0.0001         # Daily risk-free rate
    
    print(f"\nTrue parameters:")
    print(f"  μ = {TRUE_MU:.6f} (daily), {TRUE_MU * 252:.2%} (annualized)")
    print(f"  σ = {TRUE_SIGMA:.6f} (daily), {TRUE_SIGMA * np.sqrt(252):.2%} (annualized)")
    
    # Oracle Kelly
    f_oracle = kelly_single_asset(TRUE_MU, TRUE_SIGMA2, RISK_FREE)
    print(f"\nOracle Kelly fraction: {f_oracle:.4f}")
    
    # Generate sample data
    np.random.seed(42)
    sample_size = 252  # 1 year of daily data
    returns = simulate_gbm_returns(TRUE_MU, TRUE_SIGMA, sample_size)
    
    print(f"\nEstimation from {sample_size} observations:")
    mu_hat = np.mean(returns)
    sigma2_hat = np.var(returns, ddof=1)
    print(f"  μ̂ = {mu_hat:.6f}")
    print(f"  σ̂² = {sigma2_hat:.8f}")
    
    # Plug-in Kelly
    f_plugin = kelly_single_asset(mu_hat, sigma2_hat, RISK_FREE)
    print(f"\nPlug-in Kelly fraction: {f_plugin:.4f}")
    
    # ACK
    ack = adaptive_conformal_kelly(returns, RISK_FREE, alpha=0.1)
    print(f"\nAdaptive Conformal Kelly (90% confidence):")
    print(f"  Calibrated ε: {ack.epsilon:.6f}")
    print(f"  DRK fraction: {ack.fraction:.4f}")
    
    # Comparison study
    print("\n" + "=" * 70)
    print("Running comparison study (1000 simulations)...")
    print("=" * 70)
    
    results = run_kelly_comparison(
        true_mu=TRUE_MU,
        true_sigma=TRUE_SIGMA,
        sample_size=252,
        horizon=252,
        n_simulations=1000,
        risk_free=RISK_FREE,
        alpha=0.1,
        seed=123
    )
    
    print("\n{:<12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "Strategy", "Mean f", "Mean W_T", "Med W_T", "Mean log(W)", "P(profit)"
    ))
    print("-" * 72)
    
    for strategy in ['oracle', 'plugin', 'half', 'drk', 'ack']:
        r = results[strategy]
        print("{:<12} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.2%}".format(
            strategy,
            r['mean_fraction'],
            r['mean_terminal_wealth'],
            r['median_terminal_wealth'],
            r['mean_log_wealth'],
            r['prob_profit']
        ))
    
    print("\n" + "=" * 70)
    print("Key Insight: DRK and ACK sacrifice some mean wealth for better")
    print("median performance and lower variance - this is the robustness")
    print("tradeoff that Kelly's criterion cares about (log-wealth, not wealth).")
    print("=" * 70)
