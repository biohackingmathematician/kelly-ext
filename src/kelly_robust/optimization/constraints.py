"""
Constraint Classes for Portfolio Optimization.

Provides modular constraint specifications for Kelly optimization.

Author: Agna Chan
Date: December 2025
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict


class Constraint(ABC):
    """Abstract base class for portfolio constraints."""
    
    @abstractmethod
    def apply(self, weights: np.ndarray) -> np.ndarray:
        """Apply constraint to weights (projection)."""
        pass
    
    @abstractmethod
    def cvxpy_constraint(self, weights):
        """Return cvxpy constraint object."""
        pass


@dataclass
class LeverageConstraint(Constraint):
    """
    Leverage constraint: sum(|w|) ≤ leverage_limit
    
    For long-only: sum(w) ≤ leverage_limit
    """
    leverage_limit: float = 1.0
    long_only: bool = True
    
    def apply(self, weights: np.ndarray) -> np.ndarray:
        """Project weights to satisfy leverage constraint."""
        if self.long_only:
            weights = np.maximum(weights, 0)
            total = np.sum(weights)
            if total > self.leverage_limit:
                weights = weights * (self.leverage_limit / total)
        else:
            total = np.sum(np.abs(weights))
            if total > self.leverage_limit:
                weights = weights * (self.leverage_limit / total)
        return weights
    
    def cvxpy_constraint(self, weights):
        import cvxpy as cp
        if self.long_only:
            return [cp.sum(weights) <= self.leverage_limit, weights >= 0]
        else:
            return [cp.norm(weights, 1) <= self.leverage_limit]


@dataclass
class LongOnlyConstraint(Constraint):
    """Long-only constraint: w ≥ 0"""
    
    def apply(self, weights: np.ndarray) -> np.ndarray:
        return np.maximum(weights, 0)
    
    def cvxpy_constraint(self, weights):
        import cvxpy as cp
        return [weights >= 0]


@dataclass
class PositionLimitConstraint(Constraint):
    """
    Position limit: w_min ≤ w_i ≤ w_max
    """
    min_weight: float = 0.0
    max_weight: float = 1.0
    
    def apply(self, weights: np.ndarray) -> np.ndarray:
        return np.clip(weights, self.min_weight, self.max_weight)
    
    def cvxpy_constraint(self, weights):
        import cvxpy as cp
        return [weights >= self.min_weight, weights <= self.max_weight]


@dataclass
class SectorConstraint(Constraint):
    """
    Sector exposure constraint.
    
    Parameters
    ----------
    sector_map : Dict[int, str]
        Mapping from asset index to sector name
    sector_limits : Dict[str, tuple]
        Mapping from sector to (min, max) exposure
    """
    sector_map: Dict[int, str] = None
    sector_limits: Dict[str, tuple] = None
    
    def __post_init__(self):
        if self.sector_map is None:
            self.sector_map = {}
        if self.sector_limits is None:
            self.sector_limits = {}
    
    def apply(self, weights: np.ndarray) -> np.ndarray:
        """Project weights to satisfy sector constraints."""
        # Simple projection: scale sectors that exceed limits
        for sector, (min_exp, max_exp) in self.sector_limits.items():
            idx = [i for i, s in self.sector_map.items() if s == sector]
            if not idx:
                continue
            
            sector_weight = np.sum(weights[idx])
            
            if sector_weight > max_exp:
                scale = max_exp / sector_weight if sector_weight > 0 else 1
                weights[idx] *= scale
            elif sector_weight < min_exp:
                # More complex: need to increase weights
                pass  # Skip for now
        
        return weights
    
    def cvxpy_constraint(self, weights):
        import cvxpy as cp
        constraints = []
        
        for sector, (min_exp, max_exp) in self.sector_limits.items():
            idx = [i for i, s in self.sector_map.items() if s == sector]
            if not idx:
                continue
            
            sector_sum = cp.sum(weights[idx])
            constraints.append(sector_sum >= min_exp)
            constraints.append(sector_sum <= max_exp)
        
        return constraints


@dataclass
class TurnoverConstraint(Constraint):
    """
    Turnover constraint: sum(|w_new - w_old|) ≤ max_turnover
    """
    max_turnover: float = 0.5
    previous_weights: np.ndarray = None
    
    def apply(self, weights: np.ndarray) -> np.ndarray:
        """Project to satisfy turnover constraint."""
        if self.previous_weights is None:
            return weights
        
        delta = weights - self.previous_weights
        turnover = np.sum(np.abs(delta))
        
        if turnover > self.max_turnover:
            # Scale the change
            scale = self.max_turnover / turnover
            weights = self.previous_weights + scale * delta
        
        return weights
    
    def cvxpy_constraint(self, weights):
        import cvxpy as cp
        
        if self.previous_weights is None:
            return []
        
        return [cp.norm(weights - self.previous_weights, 1) <= self.max_turnover]


def apply_constraints(
    weights: np.ndarray,
    constraints: List[Constraint],
    max_iterations: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Apply multiple constraints via alternating projections.
    
    Parameters
    ----------
    weights : np.ndarray
        Initial weights
    constraints : List[Constraint]
        List of constraints to apply
    max_iterations : int
        Maximum iterations for convergence
    tol : float
        Convergence tolerance
        
    Returns
    -------
    np.ndarray
        Projected weights satisfying all constraints
    """
    for iteration in range(max_iterations):
        weights_old = weights.copy()
        
        for constraint in constraints:
            weights = constraint.apply(weights)
        
        # Check convergence
        if np.max(np.abs(weights - weights_old)) < tol:
            break
    
    return weights


def combine_cvxpy_constraints(
    weights,
    constraints: List[Constraint],
) -> list:
    """
    Combine all constraints into a single cvxpy constraint list.
    
    Parameters
    ----------
    weights : cvxpy Variable
        Optimization variable
    constraints : List[Constraint]
        Constraint objects
        
    Returns
    -------
    list
        List of cvxpy constraints
    """
    all_constraints = []
    
    for constraint in constraints:
        cvx_constraints = constraint.cvxpy_constraint(weights)
        all_constraints.extend(cvx_constraints)
    
    return all_constraints

