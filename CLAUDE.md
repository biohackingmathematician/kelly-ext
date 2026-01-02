# Development Notes

## Project Context

Research project: **Distributionally Robust Kelly Optimization**  
Target venues: Mathematical Finance, Operations Research, SIAM Journal on Financial Mathematics

## Mathematical Foundation

**Classical Kelly:** `f* = (μ - r) / σ²`

**DRK (Distributionally Robust Kelly):** `f_DRK = (μ̂ - r - ε) / σ̂²`

The ambiguity radius ε is calibrated via conformal prediction to achieve finite-sample coverage guarantees.

## Core Theorems

1. **Theorem 1:** Closed-form DRK solution under Gaussian returns with 2-Wasserstein ambiguity
2. **Theorem 2:** Conformal calibration achieves P(|μ̂ - μ| ≤ ε) ≥ 1-α
3. **Theorem 3:** Multi-asset DRK reformulates as a tractable SOCP

## Key Formulas

Growth rate: `g(f) = r + f(μ-r) - (1/2)f²σ²`

Multi-asset worst-case: `g_wc(F) = F'(μ̂-r) - (1/2)F'ΣF - ε||F||_2`

Conformal quantile: `q = ceil((1-α)(n+1))/n`

## Dependencies

Core: numpy, scipy, pandas, cvxpy  
Optional: mapie (conformal), yfinance (data), mosek (SOCP solver)

## Running Tests

```bash
pytest tests/test_kelly.py -v
```

## Quick Demo

```python
from kelly_robust import run_kelly_comparison
results = run_kelly_comparison(0.0005, 0.025, 252, 252, 1000, seed=42)
```
