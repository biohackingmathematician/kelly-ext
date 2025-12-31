"""
Data module for financial data acquisition and preprocessing.

Provides:
- Yahoo Finance data download
- Data cleaning and validation
- Return computation
- Multi-asset universe construction
"""

from .download import (
    download_prices,
    download_sp500_constituents,
    download_eurostoxx50_constituents,
    get_risk_free_rate,
)
from .preprocessing import (
    compute_returns,
    align_series,
    filter_universe,
    compute_rolling_statistics,
)

__all__ = [
    # Download
    'download_prices',
    'download_sp500_constituents',
    'download_eurostoxx50_constituents',
    'get_risk_free_rate',
    # Preprocessing
    'compute_returns',
    'align_series',
    'filter_universe',
    'compute_rolling_statistics',
]

