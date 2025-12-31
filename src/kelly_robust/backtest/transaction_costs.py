"""
Transaction Cost Models for Backtesting.

Provides various cost models:
- Fixed costs per trade
- Proportional costs (basis points)
- Tiered costs (volume-based)
- Market impact models

Author: Agna Chan
Date: December 2025
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class TransactionCostModel(ABC):
    """Abstract base class for transaction cost models."""
    
    @abstractmethod
    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
        prices: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute transaction cost for a rebalance.
        
        Parameters
        ----------
        old_weights : np.ndarray
            Current portfolio weights
        new_weights : np.ndarray
            Target portfolio weights
        portfolio_value : float
            Current portfolio value
        prices : np.ndarray, optional
            Current asset prices (for some cost models)
            
        Returns
        -------
        float
            Total transaction cost (in same units as portfolio_value)
        """
        pass
    
    def compute_turnover(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
    ) -> float:
        """Compute portfolio turnover."""
        return np.sum(np.abs(new_weights - old_weights))


@dataclass
class FixedCost(TransactionCostModel):
    """
    Fixed cost per trade.
    
    Parameters
    ----------
    cost_per_trade : float
        Fixed cost per asset traded
    """
    cost_per_trade: float = 10.0
    
    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
        prices: Optional[np.ndarray] = None,
    ) -> float:
        # Count number of assets with weight changes
        trades = np.abs(new_weights - old_weights) > 1e-8
        n_trades = np.sum(trades)
        
        return n_trades * self.cost_per_trade


@dataclass
class ProportionalCost(TransactionCostModel):
    """
    Proportional (basis point) cost model.
    
    Cost = turnover × (bps / 10000) × portfolio_value
    
    Parameters
    ----------
    cost_bps : float
        Cost in basis points (round-trip)
    """
    cost_bps: float = 10.0
    
    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
        prices: Optional[np.ndarray] = None,
    ) -> float:
        turnover = self.compute_turnover(old_weights, new_weights)
        return turnover * (self.cost_bps / 10000) * portfolio_value


@dataclass
class TieredCost(TransactionCostModel):
    """
    Tiered cost model based on trade size.
    
    Different cost rates apply to different trade size tiers.
    
    Parameters
    ----------
    tiers : list of (threshold, bps)
        List of (dollar_threshold, cost_bps) pairs.
        Costs are applied to the portion of trade in each tier.
    """
    tiers: list = None
    
    def __post_init__(self):
        if self.tiers is None:
            # Default tiers
            self.tiers = [
                (0, 10),        # 0-10k: 10 bps
                (10_000, 8),    # 10k-100k: 8 bps
                (100_000, 5),   # 100k-1M: 5 bps
                (1_000_000, 3), # 1M+: 3 bps
            ]
        # Sort by threshold
        self.tiers = sorted(self.tiers, key=lambda x: x[0])
    
    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
        prices: Optional[np.ndarray] = None,
    ) -> float:
        # Compute dollar turnover per asset
        weight_changes = np.abs(new_weights - old_weights)
        dollar_changes = weight_changes * portfolio_value
        
        total_cost = 0.0
        
        for dollar_amount in dollar_changes:
            if dollar_amount < 1e-8:
                continue
                
            remaining = dollar_amount
            prev_threshold = 0
            
            for threshold, bps in self.tiers:
                if remaining <= 0:
                    break
                    
                tier_amount = min(remaining, threshold - prev_threshold) if threshold > prev_threshold else remaining
                if tier_amount > 0:
                    total_cost += tier_amount * (bps / 10000)
                    remaining -= tier_amount
                    
                prev_threshold = threshold
            
            # Anything remaining uses the last tier rate
            if remaining > 0:
                total_cost += remaining * (self.tiers[-1][1] / 10000)
        
        return total_cost


@dataclass  
class MarketImpactCost(TransactionCostModel):
    """
    Square-root market impact model.
    
    Impact = η × σ × √(Q / ADV)
    
    where:
    - η is the impact coefficient
    - σ is volatility
    - Q is trade size
    - ADV is average daily volume
    
    Parameters
    ----------
    impact_coef : float
        Impact coefficient η (typically 0.1-0.5)
    volatility : float or np.ndarray
        Asset volatility (annualized)
    adv : float or np.ndarray
        Average daily volume in dollars
    """
    impact_coef: float = 0.1
    volatility: float = 0.2
    adv: float = 1_000_000
    
    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
        prices: Optional[np.ndarray] = None,
    ) -> float:
        # Compute dollar turnover
        weight_changes = np.abs(new_weights - old_weights)
        trade_sizes = weight_changes * portfolio_value
        
        # Handle scalar or array volatility/adv
        vol = np.atleast_1d(self.volatility)
        adv = np.atleast_1d(self.adv)
        
        if len(vol) == 1:
            vol = np.full(len(old_weights), vol[0])
        if len(adv) == 1:
            adv = np.full(len(old_weights), adv[0])
        
        # Square-root impact
        daily_vol = vol / np.sqrt(252)
        participation = trade_sizes / adv
        
        # Impact cost = η × σ_daily × √participation × trade_size
        impact_per_asset = (
            self.impact_coef * 
            daily_vol * 
            np.sqrt(np.clip(participation, 0, 1)) * 
            trade_sizes
        )
        
        return np.sum(impact_per_asset)


def create_cost_model(model_type: str = 'proportional', **kwargs) -> TransactionCostModel:
    """
    Factory function to create cost models.
    
    Parameters
    ----------
    model_type : str
        One of 'fixed', 'proportional', 'tiered', 'impact'
    **kwargs
        Parameters for the specific model
        
    Returns
    -------
    TransactionCostModel
        Instantiated cost model
    """
    models = {
        'fixed': FixedCost,
        'proportional': ProportionalCost,
        'tiered': TieredCost,
        'impact': MarketImpactCost,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](**kwargs)

