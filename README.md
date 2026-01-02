# Distributionally Robust Kelly Optimization

**Optimal Growth-Rate Portfolio Allocation Under Estimated Return Dynamics**

A mathematically rigorous framework for Kelly criterion optimization when return distributions must be estimated from data. This research provides the first principled derivation of "fractional Kelly" with finite-sample guarantees using tools from distributionally robust optimization (DRO), optimal transport, and conformal prediction.

---

## Overview

The classical Kelly criterion maximizes expected log-growth but assumes known return distributions. In practice, parameters must be estimated, leading to severe overbetting risk. This project develops **Distributionally Robust Kelly (DRK)** optimization that:

1. Accounts for parameter estimation uncertainty via Wasserstein ambiguity sets
2. Provides closed-form solutions for single-asset allocation
3. Extends to multi-asset portfolios via tractable SOCP formulations
4. Calibrates ambiguity using conformal prediction with finite-sample guarantees

### Key Results

**Theorem 1 (DRK Closed-Form):** Under 2-Wasserstein ambiguity, the optimal robust Kelly fraction is:

```
f_DRK = (μ̂ - r - ε) / σ̂²
```

where ε is the ambiguity radius, providing a principled derivation of fractional Kelly.

**Theorem 2 (Conformal Coverage):** Split conformal calibration achieves:

```
P(|μ̂ - μ| ≤ ε) ≥ 1 - α
```

This is a distribution-free guarantee that holds for any continuous distribution.

**Theorem 3 (Multi-Asset SOCP):** The multi-asset DRK problem admits a tractable convex reformulation:

```
max_f [f'(μ̂-r) - (1/2)f'Σ̂f - ε||f||₂]
```

This is a Second-Order Cone Program (SOCP), solvable in polynomial time via standard solvers.

---

## Installation

### Requirements

- Python 3.11+
- NumPy, SciPy, Pandas
- cvxpy (for multi-asset optimization)
- yfinance (for data download)

### Setup

```bash
git clone https://github.com/biohackingmathematician/kelly-ext.git
cd kelly-ext
pip install -e ".[dev,data,notebooks]"
```

Or install dependencies directly:

```bash
pip install numpy scipy pandas cvxpy matplotlib seaborn yfinance
```

---

## Quick Start

### Single-Asset DRK

```python
import numpy as np
from kelly_robust import adaptive_conformal_kelly

# Historical returns (daily)
returns = np.random.randn(252) * 0.02 + 0.0005

# Compute robust Kelly fraction with 90% confidence
result = adaptive_conformal_kelly(returns, risk_free=0.0001, alpha=0.1)

print(f"DRK fraction: {result.fraction:.4f}")
print(f"Calibrated epsilon: {result.epsilon:.6f}")
```

### Monte Carlo Comparison

```python
from kelly_robust import run_kelly_comparison

results = run_kelly_comparison(
    true_mu=0.0005,      # True mean (daily)
    true_sigma=0.02,     # True volatility
    sample_size=252,     # Estimation window
    horizon=252,         # Investment horizon
    n_simulations=1000,
    seed=42
)

for strategy in ['oracle', 'plugin', 'half', 'drk', 'ack']:
    r = results[strategy]
    print(f"{strategy:>8}: fraction={r['mean_fraction']:.4f}, "
          f"median_wealth={r['median_terminal_wealth']:.4f}")
```

### Walk-Forward Backtest

```python
from kelly_robust.backtest import WalkForwardBacktest
from kelly_robust.data import download_prices

# Download data
prices = download_prices('SPY', start='2015-01-01')
returns = prices.pct_change().dropna()

# Define strategy
def drk_strategy(hist_returns):
    from kelly_robust import adaptive_conformal_kelly
    result = adaptive_conformal_kelly(hist_returns.flatten(), alpha=0.1)
    return np.array([result.fraction])

# Run backtest
bt = WalkForwardBacktest(lookback=252, rebalance_freq=21, cost_bps=10)
result = bt.run(returns, drk_strategy)

print(f"Sharpe: {result.sharpe_ratio:.3f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
```

---

## Project Structure

```
kelly-ext/
├── src/kelly_robust/
│   ├── core/
│   │   └── kelly.py              # Core Kelly and DRK implementations
│   ├── backtest/
│   │   ├── walk_forward.py       # Walk-forward backtesting engine
│   │   ├── transaction_costs.py  # Transaction cost models
│   │   └── performance.py        # Performance metrics and statistical tests
│   ├── calibration/
│   │   ├── conformal.py          # Conformal prediction calibration
│   │   ├── bootstrap.py          # Bootstrap methods
│   │   └── bayesian.py           # Bayesian posterior calibration
│   ├── simulation/
│   │   ├── return_models.py      # GBM, Student-t, GARCH, jump-diffusion
│   │   └── monte_carlo.py        # Monte Carlo simulation engine
│   ├── data/
│   │   ├── download.py           # Yahoo Finance, FRED data download
│   │   └── preprocessing.py      # Data cleaning and transformations
│   └── optimization/
│       ├── single_asset.py       # Closed-form single-asset solutions
│       ├── multi_asset.py        # Multi-asset SOCP formulation
│       └── constraints.py        # Portfolio constraints
├── tests/
│   └── test_kelly.py             # Comprehensive unit tests
├── notebooks/
│   └── 01_drk_demo.ipynb         # Interactive demonstration
├── paper/
│   └── kelly_research_outline.md # Research outline and proofs
└── data/                         # Downloaded financial data
```

---

## Empirical Results

### S&P 500 Backtest (2016-2024)

| Strategy | Ann. Return | Max Drawdown | Sharpe |
|----------|-------------|--------------|--------|
| Buy & Hold | 14.63% | 33.72% | 0.679 |
| Plug-in Kelly | 11.44% | 33.72% | 0.571 |
| DRK (90%) | 6.29% | 12.44% | 0.506 |

**Key Finding:** DRK reduced maximum drawdown by 63% (33.72% to 12.44%) by explicitly accounting for estimation uncertainty.

### Euro Stoxx 50 Backtest (2016-2024)

| Strategy | Ann. Return | Max Drawdown | Sharpe | Turnover |
|----------|-------------|--------------|--------|----------|
| Equal Weight | 19.66% | 38.39% | 0.800 | 1.0x |
| Kelly Multi | 17.65% | 40.42% | 0.649 | 47.0x |
| DRK Multi | 19.66% | 38.39% | 0.800 | 1.0x |

**Key Finding:** DRK Multi converged to equal weight under high parameter uncertainty, avoiding the overfitting exhibited by plug-in Kelly (47x turnover).

---

## Testing

Run the test suite:

```bash
pytest tests/test_kelly.py -v
```

With coverage:

```bash
pytest tests/test_kelly.py -v --cov=src --cov-report=term-missing
```

---

## Mathematical Background

### Classical Kelly Criterion

For a risky asset with return distribution P*, the Kelly criterion maximizes:

```
g(f) = E[log(1 + r + f(R - r))]
```

For Gaussian returns, this yields f* = (μ - r) / σ².

### Distributionally Robust Formulation

When P* is unknown, we consider a worst-case over an ambiguity set:

```
f_DRK = argmax_f min_{Q ∈ A_ε} g(f; Q)
```

where A_ε is the Wasserstein ball of radius ε centered at the empirical distribution.

### Conformal Calibration

The ambiguity radius ε is calibrated using split conformal prediction:

1. Split data into training and calibration sets
2. Compute nonconformity scores on calibration set
3. Set ε as the (1-α) quantile of scores

This provides distribution-free finite-sample coverage guarantees.

---

## References

1. Kelly, J.L. (1956). "A New Interpretation of Information Rate." Bell System Technical Journal.
2. Blanchet, J. & Murthy, K. (2019). "Quantifying Distributional Model Risk via Optimal Transport." Mathematics of Operations Research.
3. Barber, R.F. et al. (2023). "Conformal Prediction Under Covariate Shift." Annals of Statistics.
4. Cover, T.M. & Thomas, J.A. (2006). Elements of Information Theory, 2nd ed. Wiley.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{chan2025drk,
  author = {Chan, Agna},
  title = {Distributionally Robust Kelly Optimization},
  year = {2025},
  url = {https://github.com/biohackingmathematician/kelly-ext}
}
```

---

## Author

**Agna Chan**  
Columbia University, Department of Statistics  
December 2025

---

## License

MIT License. See [LICENSE](LICENSE) for details.
