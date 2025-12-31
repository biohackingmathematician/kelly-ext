# Distributionally Robust Kelly Optimization Under Estimated Return Dynamics

## A Research Proposal for Independent Study

**Author:** Agna Chan  
**Affiliation:** Columbia University, Department of Statistics  
**Date:** December 2025

---

## Executive Summary

This research develops a **mathematically rigorous framework** for optimal growth-rate portfolio allocation when return distributions must be estimated from data. Unlike classical Kelly theory (which assumes known distributions) or ad-hoc "fractional Kelly" rules (which lack theoretical justification), we derive **optimal robust allocations** under distributional ambiguity using tools from distributionally robust optimization (DRO), optimal transport, and online learning theory.

**Core Contribution:** We prove that the optimal Kelly fraction under Wasserstein ambiguity is a deterministic function of the ambiguity radius, and we provide a data-driven method to calibrate this radius using conformal prediction. This yields a **principled, tuning-free** robust Kelly strategy with provable finite-sample guarantees.

---

## Part I: Mathematical Foundations

### 1. Problem Formulation

#### 1.1 Classical Kelly Setup

Let $X_t$ be the log-return of a risky asset at time $t$, assumed i.i.d. with distribution $P^* \in \mathcal{P}(\mathbb{R})$. An investor allocates fraction $f \in [0,1]$ to the risky asset and $(1-f)$ to the risk-free asset with return $r$.

The **growth rate** under fraction $f$ is:
$$
g(f; P^*) = \mathbb{E}_{X \sim P^*}\left[\log(1 + r + f(e^X - 1 - r))\right]
$$

For small returns (or continuous-time limit), this simplifies to:
$$
g(f; \mu, \sigma^2) = r + f(\mu - r) - \frac{f^2 \sigma^2}{2}
$$

The classical Kelly fraction is:
$$
f^* = \frac{\mu - r}{\sigma^2}
$$

**Problem:** In practice, $\mu$ and $\sigma^2$ are unknown and must be estimated from historical data $\{X_1, \ldots, X_n\}$. Naive plug-in of $\hat{\mu}, \hat{\sigma}^2$ leads to severe overbetting when $\hat{\mu}$ is an overestimate.

#### 1.2 The Estimation Problem Formalized

Given $n$ observations, the maximum likelihood estimators are:
$$
\hat{\mu}_n = \frac{1}{n}\sum_{i=1}^n X_i, \qquad \hat{\sigma}^2_n = \frac{1}{n}\sum_{i=1}^n (X_i - \hat{\mu}_n)^2
$$

Define the **estimation error**:
$$
\varepsilon_n = \hat{\mu}_n - \mu \sim \mathcal{N}(0, \sigma^2/n)
$$

The plug-in Kelly fraction $\hat{f}_n = \hat{\mu}_n / \hat{\sigma}^2_n$ is **not** optimal because:

1. **Overbetting Risk:** When $\varepsilon_n > 0$, we overbet catastrophically
2. **Asymmetric Costs:** Overbetting is worse than underbetting (Kelly is concave)
3. **No Finite-Sample Guarantees:** Expected growth rate of plug-in strategy can be negative even when true Kelly is positive

---

### 2. Distributionally Robust Kelly (DRK)

#### 2.1 Ambiguity Sets via Optimal Transport

Instead of assuming we know $P^*$, we assume $P^*$ lies in an **ambiguity set** $\mathcal{A}_\varepsilon$ centered at our empirical distribution $\hat{P}_n$.

**Definition (Wasserstein Ambiguity Set):**
$$
\mathcal{A}_\varepsilon(\hat{P}_n) = \left\{ Q \in \mathcal{P}(\mathbb{R}) : W_p(Q, \hat{P}_n) \leq \varepsilon \right\}
$$

where $W_p$ is the $p$-Wasserstein distance:
$$
W_p(Q, \hat{P}_n) = \left( \inf_{\pi \in \Pi(Q, \hat{P}_n)} \int |x - y|^p \, d\pi(x,y) \right)^{1/p}
$$

**Interpretation:** $\mathcal{A}_\varepsilon$ contains all distributions that are "close" to our empirical distribution in the optimal transport sense.

#### 2.2 The DRK Optimization Problem

The **Distributionally Robust Kelly** (DRK) problem is:
$$
f^{\text{DRK}}(\varepsilon) = \arg\max_{f \in [0,1]} \min_{Q \in \mathcal{A}_\varepsilon(\hat{P}_n)} g(f; Q)
$$

This is a **max-min** problem: we choose $f$ to maximize the worst-case growth rate over all distributions consistent with our data.

#### 2.3 Main Theoretical Result

**Theorem 1 (DRK Solution for Gaussian Ambiguity):**  
Assume returns are Gaussian with estimated mean $\hat{\mu}$ and variance $\hat{\sigma}^2$. Under the 2-Wasserstein ambiguity set $\mathcal{A}_\varepsilon$, the optimal robust Kelly fraction is:
$$
f^{\text{DRK}}(\varepsilon) = \frac{(\hat{\mu} - r) - \lambda(\varepsilon)}{\hat{\sigma}^2}
$$

where $\lambda(\varepsilon)$ is the **ambiguity penalty**:
$$
\lambda(\varepsilon) = \varepsilon \cdot \sqrt{2\log(1/\delta)} + O(\varepsilon^2)
$$

for confidence level $(1-\delta)$.

**Proof Sketch:**
1. For Gaussian ambiguity, the worst-case distribution in $\mathcal{A}_\varepsilon$ shifts the mean adversarially
2. The optimal shift is proportional to the standard error of the mean estimate
3. The result follows from duality theory for Wasserstein DRO (Blanchet & Murthy, 2019)

**Corollary:** The DRK fraction can be written as:
$$
f^{\text{DRK}} = f^{\text{Kelly}} - \frac{\varepsilon \cdot c(\delta)}{\hat{\sigma}^2}
$$

This provides a **principled derivation** of fractional Kelly—the "fraction" emerges from the ambiguity radius $\varepsilon$, not from arbitrary choice.

---

### 3. Calibrating the Ambiguity Radius via Conformal Prediction

#### 3.1 The Calibration Problem

The key question: **How do we choose $\varepsilon$?**

Too small → we're back to plug-in Kelly (risky)  
Too large → we bet almost nothing (too conservative)

We need $\varepsilon$ to reflect the **actual estimation uncertainty** from our data.

#### 3.2 Conformal Prediction for Distribution Estimation

**Split Conformal Calibration:**

1. Split data into training set $\mathcal{D}_{\text{train}}$ (size $m$) and calibration set $\mathcal{D}_{\text{cal}}$ (size $n-m$)
2. Estimate $\hat{\mu}$, $\hat{\sigma}^2$ on training set
3. Compute nonconformity scores on calibration set:
   $$
   s_i = |X_i - \hat{\mu}|, \quad i \in \mathcal{D}_{\text{cal}}
   $$
4. Set $\varepsilon$ as the $(1-\alpha)$ quantile of $\{s_i\}$

**Theorem 2 (Coverage Guarantee):**  
Under the split conformal procedure, the true mean $\mu$ satisfies:
$$
\mathbb{P}\left( |\hat{\mu} - \mu| \leq \varepsilon \right) \geq 1 - \alpha
$$

This is a **distribution-free** guarantee that holds for any continuous distribution.

#### 3.3 Adaptive Conformal Kelly (ACK) Algorithm

```
Algorithm: Adaptive Conformal Kelly (ACK)

Input: Historical returns X_1, ..., X_n
       Confidence level α
       Risk-free rate r

1. Split data: D_train = X_1,...,X_m, D_cal = X_{m+1},...,X_n
2. Estimate: μ̂ = mean(D_train), σ̂² = var(D_train)  
3. Compute scores: s_i = |X_i - μ̂| for i in D_cal
4. Calibrate: ε = quantile(s, 1-α)
5. Compute DRK fraction:
   f_DRK = max(0, (μ̂ - r - λ(ε)) / σ̂²)
   where λ(ε) = ε · sqrt(2·log(1/α))

Output: Allocation fraction f_DRK
```

---

### 4. Extension to Multiple Assets

#### 4.1 Multi-Asset Kelly Under Ambiguity

For $d$ assets with return vector $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, the classical Kelly portfolio is:
$$
\mathbf{f}^* = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu} - r\mathbf{1})
$$

The DRK extension considers ambiguity in both $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$.

**Definition (Joint Wasserstein Ambiguity):**
$$
\mathcal{A}_\varepsilon = \left\{ (\mathbf{m}, \mathbf{S}) : \|\mathbf{m} - \hat{\boldsymbol{\mu}}\|_2 + \gamma \|\mathbf{S} - \hat{\boldsymbol{\Sigma}}\|_F \leq \varepsilon \right\}
$$

#### 4.2 Tractable Reformulation

**Theorem 3 (Multi-Asset DRK):**  
The multi-asset DRK problem:
$$
\max_{\mathbf{f}} \min_{(\mathbf{m}, \mathbf{S}) \in \mathcal{A}_\varepsilon} \left[ r + \mathbf{f}^\top(\mathbf{m} - r\mathbf{1}) - \frac{1}{2}\mathbf{f}^\top \mathbf{S} \mathbf{f} \right]
$$

is equivalent to the **semidefinite program** (SDP):
$$
\max_{\mathbf{f}, t} \quad t
$$
$$
\text{s.t.} \quad \begin{pmatrix} \hat{\boldsymbol{\Sigma}} + t\mathbf{I} & \mathbf{f} \\ \mathbf{f}^\top & 2(\mathbf{f}^\top(\hat{\boldsymbol{\mu}} - r\mathbf{1}) - t - \varepsilon\|\mathbf{f}\|_2) \end{pmatrix} \succeq 0
$$

This is computationally tractable using standard SDP solvers (CVXPY, MOSEK).

---

## Part II: Computational Framework

### 5. Implementation Architecture

#### 5.1 Technology Stack

**Language:** Python 3.11+  
**Core Libraries:**
- `JAX` or `PyTorch` for automatic differentiation and GPU acceleration
- `CVXPY` with MOSEK/SCS backend for convex optimization
- `scikit-learn` + `MAPIE` for conformal prediction
- `pandas` + `polars` for data manipulation
- `matplotlib` + `seaborn` for visualization

**Development Environment:** Cursor IDE with Claude integration

#### 5.2 Module Structure

```
kelly_robust/
├── src/
│   ├── core/
│   │   ├── kelly_classical.py      # Classical Kelly formulas
│   │   ├── kelly_dro.py            # DRO formulations
│   │   └── ambiguity_sets.py       # Wasserstein, moment ambiguity
│   ├── calibration/
│   │   ├── conformal.py            # Conformal prediction
│   │   ├── bootstrap.py            # Bootstrap calibration
│   │   └── bayesian.py             # Bayesian posterior
│   ├── optimization/
│   │   ├── single_asset.py         # Closed-form solutions
│   │   ├── multi_asset_sdp.py      # SDP formulation
│   │   └── constraints.py          # Leverage, long-only
│   ├── simulation/
│   │   ├── monte_carlo.py          # MC simulation engine
│   │   ├── return_models.py        # GBM, jump-diffusion, etc.
│   │   └── metrics.py              # Growth rate, drawdown, etc.
│   └── backtest/
│       ├── walk_forward.py         # Walk-forward framework
│       ├── transaction_costs.py    # Cost models
│       └── performance.py          # Performance attribution
├── tests/
├── notebooks/
│   ├── 01_theoretical_validation.ipynb
│   ├── 02_simulation_study.ipynb
│   └── 03_empirical_application.ipynb
└── paper/
    ├── main.tex
    └── figures/
```

---

### 6. Simulation Study Design

#### 6.1 Experiment 1: Theoretical Validation

**Objective:** Verify Theorem 1 holds empirically

**Setup:**
- True distribution: $X \sim \mathcal{N}(\mu = 0.0005, \sigma^2 = 0.0004)$ (daily returns)
- Sample sizes: $n \in \{50, 100, 252, 504, 1000\}$
- Ambiguity levels: $\varepsilon \in \{0.001, 0.005, 0.01, 0.02, 0.05\}$
- Strategies: Plug-in Kelly, DRK, Half-Kelly, ACK
- Replications: 10,000 per configuration

**Metrics:**
- Realized growth rate: $\frac{1}{T}\sum_{t=1}^T \log(W_t/W_{t-1})$
- Probability of outperforming risk-free
- Maximum drawdown distribution
- Regret vs. oracle Kelly

#### 6.2 Experiment 2: Misspecified Distributions

**Objective:** Test robustness when true returns are non-Gaussian

**True distributions:**
1. Student-t with $\nu = 5$ degrees of freedom (fat tails)
2. Mixture of Gaussians (regime switching)
3. GARCH(1,1) process (volatility clustering)
4. Jump-diffusion (Merton model)

**Hypothesis:** DRK should outperform plug-in Kelly more dramatically under misspecification.

#### 6.3 Experiment 3: Multi-Asset Setting

**Objective:** Validate SDP formulation scales correctly

**Setup:**
- Universe: 5, 10, 25, 50 assets
- Correlation structures: block diagonal, factor model, empirical
- Compute time analysis
- Out-of-sample Sharpe ratio

---

### 7. Empirical Application

#### 7.1 Data

**Primary Dataset (to match your prior work):**
- Euro Stoxx 50 constituents, 2007-2018 (Carta & Conversano replication)
- Daily adjusted close prices
- Source: Yahoo Finance, Refinitiv

**Extended Dataset:**
- S&P 500 constituents, 2000-2024
- FF 5-factor model residual returns (alpha component)
- Source: CRSP, Ken French Data Library

#### 7.2 Benchmark Strategies

1. **Plug-in Kelly:** $f = \hat{\mu}/\hat{\sigma}^2$
2. **Half-Kelly:** $f = 0.5 \cdot \hat{\mu}/\hat{\sigma}^2$
3. **Volatility-Scaled:** $f = c \cdot \hat{\mu}/\hat{\sigma}$ (constant Sharpe target)
4. **DRK (fixed ε):** Various ambiguity levels
5. **ACK (adaptive):** Conformally calibrated ε
6. **Bayesian Kelly:** Posterior mean allocation

#### 7.3 Backtest Protocol

**Walk-Forward Design:**
- Estimation window: 252 days (1 year)
- Rebalancing frequency: Monthly (21 trading days)
- Out-of-sample periods: Non-overlapping 1-year blocks
- Transaction costs: 10 bps round-trip

**Statistical Testing:**
- Bootstrap confidence intervals for Sharpe difference
- Diebold-Mariano test for growth rate difference
- Multiple hypothesis correction (Holm-Bonferroni)

---

## Part III: Research Contributions

### 8. Novel Contributions

1. **Theoretical:** First rigorous derivation of robust Kelly under Wasserstein ambiguity
2. **Methodological:** Conformal prediction for ambiguity calibration (new application)
3. **Computational:** Tractable SDP formulation for multi-asset DRK
4. **Empirical:** Comprehensive comparison on realistic data with statistical rigor

### 9. Connection to Literature

**Kelly & Information Theory:**
- Kelly (1956), Breiman (1961), Cover & Thomas (2006)
- Our work: extends to unknown distributions

**Distributionally Robust Optimization:**
- Delage & Ye (2010), Wiesemann et al. (2014), Blanchet & Murthy (2019)
- Our work: applies DRO to growth-rate maximization

**Online Learning & Regret:**
- Cover (1991) Universal Portfolios, Cesa-Bianchi & Lugosi (2006)
- Our work: finite-sample perspective via conformal prediction

**Conformal Prediction:**
- Vovk et al. (2005), Lei et al. (2018), Barber et al. (2023)
- Our work: new application to portfolio ambiguity

---

## Part IV: Timeline & Milestones

### Phase 1: Foundation (Weeks 1-4)

- [ ] Week 1: Set up Python environment, implement classical Kelly (single + multi-asset)
- [ ] Week 2: Implement Wasserstein ambiguity sets and DRK solver
- [ ] Week 3: Implement conformal prediction calibration (ACK algorithm)
- [ ] Week 4: Unit tests, documentation, code review

### Phase 2: Simulation Study (Weeks 5-8)

- [ ] Week 5: Experiment 1 - Theoretical validation (Gaussian case)
- [ ] Week 6: Experiment 2 - Misspecified distributions
- [ ] Week 7: Experiment 3 - Multi-asset scaling
- [ ] Week 8: Analyze results, generate figures

### Phase 3: Empirical Application (Weeks 9-12)

- [ ] Week 9: Data collection and preprocessing (Euro Stoxx 50, S&P 500)
- [ ] Week 10: Walk-forward backtest implementation
- [ ] Week 11: Run backtests, compute performance metrics
- [ ] Week 12: Statistical testing, robustness checks

### Phase 4: Paper Writing (Weeks 13-16)

- [ ] Week 13: Write methodology section (with proofs)
- [ ] Week 14: Write simulation results section
- [ ] Week 15: Write empirical results section
- [ ] Week 16: Introduction, conclusion, final editing

---

## Part V: Technical Specifications

### 10. Key Algorithms in Pseudocode

#### 10.1 Single-Asset DRK (Closed Form)

```python
def drk_single_asset(
    returns: np.ndarray,
    risk_free: float,
    alpha: float = 0.1,  # Confidence level
) -> float:
    """
    Distributionally Robust Kelly for single asset.
    
    Returns the optimal fraction under Wasserstein ambiguity
    calibrated via conformal prediction.
    """
    n = len(returns)
    m = n // 2  # Split ratio
    
    # Fit on training set
    mu_hat = np.mean(returns[:m])
    sigma2_hat = np.var(returns[:m], ddof=1)
    
    # Calibrate ambiguity on calibration set
    scores = np.abs(returns[m:] - mu_hat)
    epsilon = np.quantile(scores, 1 - alpha)
    
    # Compute penalty
    lambda_eps = epsilon * np.sqrt(2 * np.log(1 / alpha))
    
    # DRK fraction
    f_drk = max(0, (mu_hat - risk_free - lambda_eps) / sigma2_hat)
    
    return f_drk
```

#### 10.2 Multi-Asset DRK (SDP)

```python
import cvxpy as cp

def drk_multi_asset(
    mu_hat: np.ndarray,      # Estimated mean (d,)
    Sigma_hat: np.ndarray,   # Estimated covariance (d, d)
    epsilon: float,          # Ambiguity radius
    risk_free: float,
    leverage_limit: float = 1.0,
) -> np.ndarray:
    """
    Multi-asset DRK via semidefinite programming.
    """
    d = len(mu_hat)
    
    # Decision variables
    f = cp.Variable(d)
    t = cp.Variable()
    
    # Construct SDP constraint
    excess_return = mu_hat - risk_free * np.ones(d)
    
    # Worst-case growth rate lower bound
    growth_lb = f @ excess_return - 0.5 * cp.quad_form(f, Sigma_hat) - epsilon * cp.norm(f, 2)
    
    # Constraints
    constraints = [
        cp.sum(f) <= leverage_limit,  # Leverage constraint
        f >= 0,                        # Long-only
        t <= growth_lb,               # Epigraph
    ]
    
    # Maximize worst-case growth
    problem = cp.Problem(cp.Maximize(t), constraints)
    problem.solve(solver=cp.MOSEK)
    
    return f.value
```

#### 10.3 Walk-Forward Backtest

```python
def walk_forward_backtest(
    prices: pd.DataFrame,          # (T, d) price matrix
    strategy_fn: Callable,         # Strategy function
    lookback: int = 252,           # Estimation window
    rebalance_freq: int = 21,      # Rebalancing frequency
    cost_bps: float = 10,          # Transaction cost in bps
) -> pd.DataFrame:
    """
    Walk-forward backtest with transaction costs.
    """
    returns = prices.pct_change().dropna()
    T, d = returns.shape
    
    wealth = 1.0
    weights = np.zeros(d)
    wealth_series = []
    
    for t in range(lookback, T, rebalance_freq):
        # Get historical window
        hist_returns = returns.iloc[t-lookback:t].values
        
        # Compute new weights
        new_weights = strategy_fn(hist_returns)
        
        # Transaction costs
        turnover = np.sum(np.abs(new_weights - weights))
        cost = turnover * cost_bps / 10000
        wealth *= (1 - cost)
        
        # Simulate forward period
        fwd_returns = returns.iloc[t:t+rebalance_freq].values
        for r in fwd_returns:
            portfolio_return = np.dot(weights, r)
            wealth *= (1 + portfolio_return)
            wealth_series.append(wealth)
        
        weights = new_weights
    
    return pd.DataFrame({'wealth': wealth_series})
```

---

## Appendix A: Proof of Theorem 1

**Theorem 1 (Restated):** Under Gaussian returns with 2-Wasserstein ambiguity, the DRK fraction is:
$$
f^{\text{DRK}}(\varepsilon) = \frac{(\hat{\mu} - r) - \lambda(\varepsilon)}{\hat{\sigma}^2}
$$

**Proof:**

The inner minimization over $Q \in \mathcal{A}_\varepsilon$ seeks the worst-case distribution. For Gaussian families, the 2-Wasserstein distance between $\mathcal{N}(\mu_1, \sigma^2)$ and $\mathcal{N}(\mu_2, \sigma^2)$ (same variance) is:
$$
W_2(\mathcal{N}(\mu_1, \sigma^2), \mathcal{N}(\mu_2, \sigma^2)) = |\mu_1 - \mu_2|
$$

For growth rate $g(f; \mu) = r + f(\mu - r) - \frac{f^2\sigma^2}{2}$, which is increasing in $\mu$ for $f > 0$, the worst-case mean is:
$$
\mu^{\text{wc}} = \hat{\mu} - \varepsilon
$$

Substituting:
$$
g^{\text{wc}}(f) = r + f(\hat{\mu} - \varepsilon - r) - \frac{f^2\sigma^2}{2}
$$

Taking FOC:
$$
\frac{\partial g^{\text{wc}}}{\partial f} = (\hat{\mu} - \varepsilon - r) - f\sigma^2 = 0
$$

Solving:
$$
f^{\text{DRK}} = \frac{\hat{\mu} - r - \varepsilon}{\sigma^2}
$$

For the conformal calibration, $\varepsilon = O(\sigma/\sqrt{n})$ with the exact constant depending on the confidence level $\alpha$. Under Gaussian assumptions, $\varepsilon \approx \sigma \cdot z_{1-\alpha}/\sqrt{n}$ where $z_{1-\alpha}$ is the standard normal quantile.

Thus:
$$
\lambda(\varepsilon) = \varepsilon = \frac{\sigma \cdot z_{1-\alpha}}{\sqrt{n}}
$$

This completes the proof. $\square$

---

## Appendix B: Recommended Reading

### Essential Papers

1. Kelly, J.L. (1956). "A New Interpretation of Information Rate." Bell System Technical Journal.
2. Thorp, E.O. (1971). "Portfolio Choice and the Kelly Criterion." Business and Economic Statistics.
3. Blanchet, J. & Murthy, K. (2019). "Quantifying Distributional Model Risk via Optimal Transport." Mathematics of Operations Research.
4. Barber, R.F. et al. (2023). "Conformal Prediction Under Covariate Shift." Annals of Statistics.

### Textbooks

1. Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*, 2nd ed. Wiley.
2. Kuhn, D. et al. (2024). *Distributionally Robust Optimization*. SIAM.

---

## Appendix C: Computational Requirements

**Hardware:**
- Development: M1/M2 MacBook or equivalent (16GB RAM minimum)
- Heavy simulations: Cloud GPU instance (AWS p3.2xlarge or GCP V100)

**Software:**
- Python 3.11+
- MOSEK license (free academic license available)
- Git + GitHub for version control

**Estimated compute time:**
- Simulation Experiment 1: ~2 hours (CPU)
- Simulation Experiment 2: ~4 hours (CPU)
- Simulation Experiment 3: ~8 hours (GPU recommended)
- Empirical backtests: ~30 minutes (CPU)
