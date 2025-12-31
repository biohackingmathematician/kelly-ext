"""
Unit tests for Distributionally Robust Kelly (DRK) implementation.

Tests cover:
1. Classical Kelly formulas (single + multi-asset)
2. Growth rate computations
3. DRK closed-form solution (Theorem 1)
4. Conformal calibration (Theorem 2)
5. Multi-asset SDP formulation (Theorem 3)
6. Monte Carlo simulation correctness
7. Edge cases and input validation

Author: Agna Chan
Date: December 2025
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

import sys
sys.path.insert(0, 'src')

from kelly_robust.core.kelly import (
    kelly_single_asset,
    kelly_multi_asset,
    growth_rate_gaussian,
    growth_rate_exact,
    drk_single_asset_closed_form,
    calibrate_epsilon_conformal,
    adaptive_conformal_kelly,
    drk_multi_asset_sdp,
    simulate_gbm_returns,
    simulate_wealth_paths,
    compute_growth_metrics,
    run_kelly_comparison,
    DRKResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_params():
    """Simple test parameters."""
    return {
        'mu': 0.001,        # 0.1% daily return
        'sigma2': 0.0004,   # 2% daily volatility
        'risk_free': 0.0001
    }


@pytest.fixture
def multi_asset_params():
    """Multi-asset test parameters."""
    np.random.seed(42)
    d = 3
    mu = np.array([0.001, 0.0015, 0.0008])
    # Create positive definite covariance matrix
    A = np.random.randn(d, d) * 0.01
    Sigma = A @ A.T + 0.0001 * np.eye(d)
    return {'mu': mu, 'Sigma': Sigma, 'risk_free': 0.0001}


@pytest.fixture
def sample_returns():
    """Generate sample returns for testing."""
    np.random.seed(123)
    return np.random.randn(252) * 0.02 + 0.0005


# =============================================================================
# Part 1: Classical Kelly Tests
# =============================================================================

class TestKellySingleAsset:
    """Tests for single-asset Kelly formula."""
    
    def test_basic_formula(self, simple_params):
        """Test f* = (μ - r) / σ²."""
        mu = simple_params['mu']
        sigma2 = simple_params['sigma2']
        r = simple_params['risk_free']
        
        f = kelly_single_asset(mu, sigma2, r)
        expected = (mu - r) / sigma2
        
        assert_allclose(f, expected, rtol=1e-10)
    
    def test_zero_excess_return(self):
        """When μ = r, Kelly fraction should be 0."""
        f = kelly_single_asset(mu=0.001, sigma2=0.01, risk_free=0.001)
        assert_allclose(f, 0.0, atol=1e-10)
    
    def test_negative_excess_return(self):
        """When μ < r, Kelly fraction should be negative (short)."""
        f = kelly_single_asset(mu=0.001, sigma2=0.01, risk_free=0.002)
        assert f < 0
    
    def test_high_volatility_reduces_fraction(self):
        """Higher volatility should reduce Kelly fraction."""
        f_low_vol = kelly_single_asset(0.01, 0.01, 0.0)
        f_high_vol = kelly_single_asset(0.01, 0.04, 0.0)
        
        assert f_high_vol < f_low_vol
    
    def test_zero_variance_raises(self):
        """Should raise ValueError for zero variance."""
        with pytest.raises(ValueError, match="Variance must be positive"):
            kelly_single_asset(0.01, 0.0, 0.0)
    
    def test_negative_variance_raises(self):
        """Should raise ValueError for negative variance."""
        with pytest.raises(ValueError, match="Variance must be positive"):
            kelly_single_asset(0.01, -0.01, 0.0)


class TestKellyMultiAsset:
    """Tests for multi-asset Kelly formula."""
    
    def test_basic_formula(self, multi_asset_params):
        """Test F* = Σ⁻¹(μ - r·1)."""
        mu = multi_asset_params['mu']
        Sigma = multi_asset_params['Sigma']
        r = multi_asset_params['risk_free']
        
        F = kelly_multi_asset(mu, Sigma, r)
        
        # Verify: Σ·F = μ - r
        excess = mu - r * np.ones(len(mu))
        reconstructed = Sigma @ F
        
        assert_allclose(reconstructed, excess, rtol=1e-10)
    
    def test_single_asset_consistency(self, simple_params):
        """Multi-asset with d=1 should match single-asset."""
        mu = np.array([simple_params['mu']])
        Sigma = np.array([[simple_params['sigma2']]])
        r = simple_params['risk_free']
        
        f_multi = kelly_multi_asset(mu, Sigma, r)[0]
        f_single = kelly_single_asset(simple_params['mu'], 
                                       simple_params['sigma2'], r)
        
        assert_allclose(f_multi, f_single, rtol=1e-10)
    
    def test_uncorrelated_assets(self):
        """Uncorrelated assets should have independent Kelly fractions."""
        mu = np.array([0.002, 0.001])
        Sigma = np.diag([0.01, 0.02])  # Diagonal = uncorrelated
        
        F = kelly_multi_asset(mu, Sigma, 0.0)
        
        # Each should equal single-asset Kelly
        assert_allclose(F[0], 0.002 / 0.01, rtol=1e-10)
        assert_allclose(F[1], 0.001 / 0.02, rtol=1e-10)


# =============================================================================
# Part 2: Growth Rate Tests
# =============================================================================

class TestGrowthRate:
    """Tests for growth rate computations."""
    
    def test_gaussian_formula(self, simple_params):
        """Test g(f) = r + f(μ-r) - (1/2)f²σ²."""
        mu = simple_params['mu']
        sigma2 = simple_params['sigma2']
        r = simple_params['risk_free']
        f = 0.5
        
        g = growth_rate_gaussian(f, mu, sigma2, r)
        expected = r + f * (mu - r) - 0.5 * f**2 * sigma2
        
        assert_allclose(g, expected, rtol=1e-10)
    
    def test_optimal_at_kelly(self, simple_params):
        """Growth rate should be maximized at Kelly fraction."""
        mu = simple_params['mu']
        sigma2 = simple_params['sigma2']
        r = simple_params['risk_free']
        
        f_kelly = kelly_single_asset(mu, sigma2, r)
        g_kelly = growth_rate_gaussian(f_kelly, mu, sigma2, r)
        
        # Test nearby points have lower growth
        for delta in [-0.1, -0.01, 0.01, 0.1]:
            g_other = growth_rate_gaussian(f_kelly + delta, mu, sigma2, r)
            assert g_other <= g_kelly + 1e-10
    
    def test_zero_fraction_gives_risk_free(self, simple_params):
        """With f=0, growth rate should equal risk-free rate."""
        r = simple_params['risk_free']
        g = growth_rate_gaussian(0, simple_params['mu'], 
                                  simple_params['sigma2'], r)
        assert_allclose(g, r, rtol=1e-10)
    
    def test_exact_vs_gaussian_small_returns(self):
        """Exact and Gaussian should agree for small returns."""
        mu = 0.0005  # Small daily return
        sigma2 = 0.0004
        f = 0.5
        
        g_gauss = growth_rate_gaussian(f, mu, sigma2, 0.0)
        g_exact = growth_rate_exact(f, mu, sigma2, 0.0)
        
        # Should be close for small returns
        assert_allclose(g_gauss, g_exact, rtol=0.05)


# =============================================================================
# Part 3: DRK Closed-Form Tests (Theorem 1)
# =============================================================================

class TestDRKClosedForm:
    """Tests for DRK closed-form solution."""
    
    def test_zero_epsilon_equals_plugin(self, simple_params):
        """With ε=0, DRK should equal plug-in Kelly."""
        mu_hat = simple_params['mu']
        sigma2_hat = simple_params['sigma2']
        r = simple_params['risk_free']
        
        f_drk = drk_single_asset_closed_form(mu_hat, sigma2_hat, 0.0, r,
                                              min_fraction=-np.inf, 
                                              max_fraction=np.inf)
        f_plugin = kelly_single_asset(mu_hat, sigma2_hat, r)
        
        assert_allclose(f_drk, f_plugin, rtol=1e-10)
    
    def test_positive_epsilon_reduces_fraction(self, simple_params):
        """Positive ε should reduce Kelly fraction."""
        mu_hat = simple_params['mu']
        sigma2_hat = simple_params['sigma2']
        r = simple_params['risk_free']
        
        f_plugin = kelly_single_asset(mu_hat, sigma2_hat, r)
        f_drk = drk_single_asset_closed_form(mu_hat, sigma2_hat, 0.0002, r,
                                              min_fraction=-np.inf,
                                              max_fraction=np.inf)
        
        assert f_drk < f_plugin
    
    def test_large_epsilon_gives_zero(self, simple_params):
        """Large ε should drive fraction to zero (with constraint)."""
        mu_hat = simple_params['mu']
        sigma2_hat = simple_params['sigma2']
        r = simple_params['risk_free']
        
        # Large epsilon makes worst-case mean negative
        f_drk = drk_single_asset_closed_form(mu_hat, sigma2_hat, 0.01, r,
                                              min_fraction=0.0, max_fraction=1.0)
        
        assert f_drk == 0.0
    
    def test_formula_correctness(self, simple_params):
        """Test f_DRK = (μ̂ - r - ε) / σ̂²."""
        mu_hat = simple_params['mu']
        sigma2_hat = simple_params['sigma2']
        r = simple_params['risk_free']
        epsilon = 0.0001
        
        f_drk = drk_single_asset_closed_form(mu_hat, sigma2_hat, epsilon, r,
                                              min_fraction=-np.inf,
                                              max_fraction=np.inf)
        expected = (mu_hat - r - epsilon) / sigma2_hat
        
        assert_allclose(f_drk, expected, rtol=1e-10)
    
    def test_constraints_applied(self):
        """Test min/max constraints are enforced."""
        f_drk = drk_single_asset_closed_form(
            mu_hat=0.01, sigma2_hat=0.001, epsilon=0.0,
            risk_free=0.0, min_fraction=0.0, max_fraction=0.5
        )
        assert f_drk == 0.5  # Would be 10 without constraint
        
        f_drk2 = drk_single_asset_closed_form(
            mu_hat=0.001, sigma2_hat=0.001, epsilon=0.005,
            risk_free=0.0, min_fraction=0.0, max_fraction=1.0
        )
        assert f_drk2 == 0.0  # Negative excess, clipped to 0


# =============================================================================
# Part 4: Conformal Calibration Tests (Theorem 2)
# =============================================================================

class TestConformalCalibration:
    """Tests for conformal prediction calibration."""
    
    def test_returns_three_values(self, sample_returns):
        """Should return epsilon, mu_hat, sigma2_hat."""
        epsilon, mu_hat, sigma2_hat = calibrate_epsilon_conformal(sample_returns)
        
        assert isinstance(epsilon, float)
        assert isinstance(mu_hat, float)
        assert isinstance(sigma2_hat, float)
    
    def test_epsilon_positive(self, sample_returns):
        """Epsilon should be non-negative."""
        epsilon, _, _ = calibrate_epsilon_conformal(sample_returns)
        assert epsilon >= 0
    
    def test_higher_alpha_lower_epsilon(self, sample_returns):
        """Higher α (lower confidence) should give smaller ε."""
        eps_90, _, _ = calibrate_epsilon_conformal(sample_returns, alpha=0.1)
        eps_50, _, _ = calibrate_epsilon_conformal(sample_returns, alpha=0.5)
        
        assert eps_50 < eps_90
    
    def test_coverage_empirical(self):
        """Verify coverage guarantee holds empirically."""
        np.random.seed(42)
        
        true_mu = 0.001
        true_sigma = 0.02
        n_trials = 500
        alpha = 0.1
        
        covered = 0
        for _ in range(n_trials):
            # Generate data
            returns = np.random.randn(200) * true_sigma + true_mu
            epsilon, mu_hat, _ = calibrate_epsilon_conformal(returns, alpha=alpha)
            
            # Check if true mean is within interval
            if abs(mu_hat - true_mu) <= epsilon:
                covered += 1
        
        coverage = covered / n_trials
        # Should be at least (1 - alpha) = 0.9
        # Allow some slack for finite sample
        assert coverage >= (1 - alpha) - 0.05
    
    def test_split_ratio_affects_estimates(self, sample_returns):
        """Different split ratios should affect estimates."""
        _, mu_hat1, _ = calibrate_epsilon_conformal(sample_returns, split_ratio=0.3)
        _, mu_hat2, _ = calibrate_epsilon_conformal(sample_returns, split_ratio=0.7)
        
        # Different training sets → different estimates
        # (Though with same data, this is deterministic)
        assert mu_hat1 != mu_hat2


class TestAdaptiveConformalKelly:
    """Tests for the ACK algorithm."""
    
    def test_returns_drk_result(self, sample_returns):
        """Should return a DRKResult dataclass."""
        result = adaptive_conformal_kelly(sample_returns)
        
        assert isinstance(result, DRKResult)
        assert hasattr(result, 'fraction')
        assert hasattr(result, 'epsilon')
        assert hasattr(result, 'confidence_level')
    
    def test_fraction_non_negative(self, sample_returns):
        """Fraction should be >= 0 with default constraints."""
        result = adaptive_conformal_kelly(sample_returns)
        assert result.fraction >= 0
    
    def test_fraction_respects_max(self, sample_returns):
        """Fraction should respect max_fraction constraint."""
        result = adaptive_conformal_kelly(sample_returns, max_fraction=0.3)
        assert result.fraction <= 0.3
    
    def test_higher_confidence_lower_fraction(self, sample_returns):
        """Higher confidence (lower α) should give lower fraction."""
        result_90 = adaptive_conformal_kelly(sample_returns, alpha=0.1)
        result_50 = adaptive_conformal_kelly(sample_returns, alpha=0.5)
        
        # Lower alpha = higher confidence = larger epsilon = lower fraction
        assert result_90.fraction <= result_50.fraction


# =============================================================================
# Part 5: Multi-Asset SDP Tests (Theorem 3)
# =============================================================================

class TestDRKMultiAssetSDP:
    """Tests for multi-asset DRK SDP formulation."""
    
    def test_zero_epsilon_approximates_classical(self, multi_asset_params):
        """With ε≈0, should approximate classical Kelly."""
        pytest.importorskip("cvxpy")
        
        mu = multi_asset_params['mu']
        Sigma = multi_asset_params['Sigma']
        r = multi_asset_params['risk_free']
        
        # Very small epsilon
        F_drk = drk_multi_asset_sdp(mu, Sigma, epsilon=1e-8, risk_free=r,
                                     leverage_limit=10.0, long_only=False)
        F_kelly = kelly_multi_asset(mu, Sigma, r)
        
        # Should be close (not exact due to numerical issues)
        assert_allclose(F_drk, F_kelly, rtol=0.1)
    
    def test_positive_epsilon_reduces_allocation(self, multi_asset_params):
        """Positive ε should reduce total allocation."""
        pytest.importorskip("cvxpy")
        
        mu = multi_asset_params['mu']
        Sigma = multi_asset_params['Sigma']
        r = multi_asset_params['risk_free']
        
        F_low = drk_multi_asset_sdp(mu, Sigma, epsilon=0.001, risk_free=r,
                                     leverage_limit=10.0, long_only=True)
        F_high = drk_multi_asset_sdp(mu, Sigma, epsilon=0.01, risk_free=r,
                                      leverage_limit=10.0, long_only=True)
        
        # Higher epsilon should give lower allocation
        assert np.sum(F_high) <= np.sum(F_low) + 1e-6
    
    def test_long_only_constraint(self, multi_asset_params):
        """With long_only=True, all weights should be >= 0."""
        pytest.importorskip("cvxpy")
        
        mu = multi_asset_params['mu']
        Sigma = multi_asset_params['Sigma']
        
        F = drk_multi_asset_sdp(mu, Sigma, epsilon=0.001, 
                                 long_only=True)
        
        assert np.all(F >= -1e-6)  # Allow small numerical errors
    
    def test_leverage_constraint(self, multi_asset_params):
        """Sum of weights should respect leverage limit."""
        pytest.importorskip("cvxpy")
        
        mu = multi_asset_params['mu']
        Sigma = multi_asset_params['Sigma']
        
        for lev in [0.5, 1.0, 2.0]:
            F = drk_multi_asset_sdp(mu, Sigma, epsilon=0.001,
                                     leverage_limit=lev, long_only=True)
            assert np.sum(F) <= lev + 1e-6


# =============================================================================
# Part 6: Simulation Tests
# =============================================================================

class TestGBMSimulation:
    """Tests for GBM return simulation."""
    
    def test_output_shape(self):
        """Should return correct shape."""
        returns = simulate_gbm_returns(0.001, 0.02, n_periods=100, n_paths=50)
        assert returns.shape == (50, 100)
    
    def test_single_path_1d(self):
        """Single path should return 1D array."""
        returns = simulate_gbm_returns(0.001, 0.02, n_periods=100, n_paths=1)
        assert returns.ndim == 1
        assert len(returns) == 100
    
    def test_seed_reproducibility(self):
        """Same seed should give same results."""
        r1 = simulate_gbm_returns(0.001, 0.02, 100, seed=42)
        r2 = simulate_gbm_returns(0.001, 0.02, 100, seed=42)
        
        assert_allclose(r1, r2)
    
    def test_mean_approximately_correct(self):
        """Sample mean should approximate true mean."""
        true_mu = 0.001
        true_sigma = 0.02
        
        # Large sample
        np.random.seed(42)
        returns = simulate_gbm_returns(true_mu, true_sigma, 
                                        n_periods=10000, n_paths=100)
        
        # GBM log-returns have mean (μ - σ²/2)
        expected_mean = true_mu - 0.5 * true_sigma**2
        sample_mean = np.mean(returns)
        
        assert_allclose(sample_mean, expected_mean, rtol=0.1)
    
    def test_std_approximately_correct(self):
        """Sample std should approximate true sigma."""
        true_sigma = 0.02
        
        np.random.seed(42)
        returns = simulate_gbm_returns(0.001, true_sigma,
                                        n_periods=10000, n_paths=100)
        sample_std = np.std(returns)
        
        assert_allclose(sample_std, true_sigma, rtol=0.05)


class TestWealthSimulation:
    """Tests for wealth path simulation."""
    
    def test_output_shape(self):
        """Should return (n_paths, T+1) array."""
        returns = np.random.randn(50, 100) * 0.02
        wealth = simulate_wealth_paths(returns, fraction=0.5)
        
        assert wealth.shape == (50, 101)
    
    def test_initial_wealth(self):
        """First column should be initial wealth."""
        returns = np.random.randn(10, 50) * 0.02
        wealth = simulate_wealth_paths(returns, fraction=0.5, initial_wealth=100)
        
        assert np.all(wealth[:, 0] == 100)
    
    def test_zero_fraction_grows_at_risk_free(self):
        """With f=0, wealth should grow at risk-free rate."""
        returns = np.random.randn(100) * 0.02 + 0.001
        risk_free = 0.0001
        
        wealth = simulate_wealth_paths(returns, fraction=0.0, 
                                        risk_free=risk_free, initial_wealth=1.0)
        
        # Should grow at exactly (1 + r)^T
        expected_final = (1 + risk_free) ** len(returns)
        assert_allclose(wealth[-1], expected_final, rtol=1e-10)
    
    def test_full_fraction_follows_returns(self):
        """With f=1 and r=0, should follow risky asset."""
        np.random.seed(42)
        log_returns = np.array([0.01, -0.02, 0.015, -0.005])
        
        wealth = simulate_wealth_paths(log_returns, fraction=1.0, risk_free=0.0)
        
        # Manually compute
        simple_returns = np.exp(log_returns) - 1
        expected = np.cumprod(1 + simple_returns)
        expected = np.insert(expected, 0, 1.0)
        
        assert_allclose(wealth, expected, rtol=1e-10)


class TestGrowthMetrics:
    """Tests for growth metric computation."""
    
    def test_all_keys_present(self):
        """Should return all expected metrics."""
        wealth = np.random.rand(100, 252) + 0.5  # Random wealth paths
        wealth = np.cumsum(wealth, axis=1)  # Make it increasing-ish
        
        metrics = compute_growth_metrics(wealth)
        
        expected_keys = [
            'mean_terminal_wealth', 'median_terminal_wealth', 'std_terminal_wealth',
            'mean_log_wealth', 'median_log_wealth',
            'mean_growth_rate', 'median_growth_rate',
            'prob_profit', 'prob_double', 'prob_ruin',
            'mean_max_drawdown', 'median_max_drawdown', 'max_max_drawdown'
        ]
        
        for key in expected_keys:
            assert key in metrics
    
    def test_prob_profit_range(self):
        """Probability should be in [0, 1]."""
        wealth = np.random.rand(100, 50) + 0.5
        wealth = np.cumsum(wealth, axis=1)
        metrics = compute_growth_metrics(wealth)
        
        assert 0 <= metrics['prob_profit'] <= 1
        assert 0 <= metrics['prob_double'] <= 1
        assert 0 <= metrics['prob_ruin'] <= 1
    
    def test_drawdown_non_negative(self):
        """Drawdowns should be non-negative."""
        wealth = np.random.rand(50, 100) + 0.5
        wealth = np.cumsum(wealth, axis=1)
        metrics = compute_growth_metrics(wealth)
        
        assert metrics['mean_max_drawdown'] >= 0
        assert metrics['max_max_drawdown'] >= 0


# =============================================================================
# Part 7: Integration Tests
# =============================================================================

class TestKellyComparison:
    """Integration tests for the comparison framework."""
    
    def test_all_strategies_present(self):
        """Should return results for all strategies."""
        results = run_kelly_comparison(
            true_mu=0.0005, true_sigma=0.02,
            sample_size=100, horizon=50,
            n_simulations=10, seed=42
        )
        
        for strategy in ['oracle', 'plugin', 'half', 'drk', 'ack']:
            assert strategy in results
    
    def test_oracle_has_fixed_fraction(self):
        """Oracle should have zero std in fraction."""
        results = run_kelly_comparison(
            true_mu=0.0005, true_sigma=0.02,
            sample_size=100, horizon=50,
            n_simulations=50, seed=42
        )
        
        # Oracle knows true params → same fraction every time
        assert results['oracle']['std_fraction'] < 1e-10
    
    def test_drk_leq_plugin_fraction(self):
        """DRK fraction should be ≤ plug-in on average."""
        results = run_kelly_comparison(
            true_mu=0.001, true_sigma=0.02,
            sample_size=100, horizon=50,
            n_simulations=100, seed=42
        )
        
        assert results['drk']['mean_fraction'] <= results['plugin']['mean_fraction'] + 0.01
    
    def test_reproducibility(self):
        """Same seed should give same results."""
        r1 = run_kelly_comparison(0.0005, 0.02, 100, 50, 20, seed=123)
        r2 = run_kelly_comparison(0.0005, 0.02, 100, 50, 20, seed=123)
        
        assert_allclose(r1['plugin']['mean_fraction'], 
                        r2['plugin']['mean_fraction'])


# =============================================================================
# Part 8: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_very_small_sample(self):
        """Should handle small samples (with warning)."""
        returns = np.array([0.01, -0.01, 0.02])
        
        with pytest.warns(UserWarning, match="Training set too small"):
            epsilon, mu_hat, sigma2_hat = calibrate_epsilon_conformal(
                returns, split_ratio=0.5
            )
    
    def test_constant_returns(self):
        """Should handle constant returns (zero variance edge case)."""
        returns = np.ones(100) * 0.001
        
        # This will have zero variance on training set
        # The function should still work (variance = 0 after ddof=1)
        epsilon, mu_hat, sigma2_hat = calibrate_epsilon_conformal(returns)
        
        assert mu_hat == 0.001
        # Variance might be 0 or near-0
    
    def test_negative_returns_mean(self):
        """Should handle negative mean returns (no profitable bet)."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02 - 0.005  # Negative mean
        
        result = adaptive_conformal_kelly(returns, risk_free=0.0)
        
        # With negative mean and positive epsilon, fraction should be 0
        assert result.fraction == 0.0
    
    def test_very_high_volatility(self):
        """Should handle extreme volatility."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.5 + 0.001  # 50% daily vol (!!)
        
        result = adaptive_conformal_kelly(returns, risk_free=0.0)
        
        # Should not crash, fraction might be 0 or very small
        assert result.fraction >= 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

