"""
Backtesting module for Kelly strategies.

This module provides:
- Walk-forward backtesting framework
- Transaction cost models
- Performance attribution and metrics
"""

from .walk_forward import (
    WalkForwardBacktest,
    BacktestResult,
    run_walk_forward_backtest,
)
from .transaction_costs import (
    TransactionCostModel,
    FixedCost,
    ProportionalCost,
    TieredCost,
)
from .performance import (
    compute_performance_metrics,
    compute_drawdown_series,
    statistical_tests,
)

__all__ = [
    # Walk-forward
    'WalkForwardBacktest',
    'BacktestResult',
    'run_walk_forward_backtest',
    # Transaction costs
    'TransactionCostModel',
    'FixedCost', 
    'ProportionalCost',
    'TieredCost',
    # Performance
    'compute_performance_metrics',
    'compute_drawdown_series',
    'statistical_tests',
]

