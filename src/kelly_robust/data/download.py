"""
Financial Data Download Utilities.

Provides functions to download price data from various sources:
- Yahoo Finance (yfinance)
- FRED (risk-free rates)

Author: Agna Chan
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union
from datetime import datetime, timedelta
import warnings


def download_prices(
    tickers: Union[str, List[str]],
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = '1d',
) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.
    
    Parameters
    ----------
    tickers : str or List[str]
        Ticker symbol(s) to download
    start : str, optional
        Start date (YYYY-MM-DD). Default: 5 years ago
    end : str, optional
        End date (YYYY-MM-DD). Default: today
    interval : str
        Data interval: '1d', '1wk', '1mo'
        
    Returns
    -------
    pd.DataFrame
        Adjusted close prices, columns = tickers, index = date
        
    Examples
    --------
    >>> prices = download_prices(['AAPL', 'MSFT'], start='2020-01-01')
    >>> prices = download_prices('SPY', start='2015-01-01', end='2023-12-31')
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance required for data download. "
            "Install via: pip install yfinance"
        )
    
    # Default date range
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    if start is None:
        start = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    # Ensure list format
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # Download data
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=True,  # Use adjusted prices
    )
    
    # Extract close prices
    # Handle different yfinance return formats
    if 'Close' in data.columns:
        prices = data['Close']
    elif isinstance(data.columns, pd.MultiIndex):
        # New yfinance format with MultiIndex columns
        prices = data.xs('Close', axis=1, level=0) if 'Close' in data.columns.get_level_values(0) else data['Close']
    else:
        prices = data
    
    # Ensure DataFrame format with proper column names
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    elif len(tickers) == 1 and len(prices.columns) == 1:
        prices.columns = tickers
    
    # Clean up
    prices = prices.dropna(how='all')
    
    return prices


def download_sp500_constituents(
    as_of_date: Optional[str] = None,
) -> List[str]:
    """
    Get S&P 500 constituent tickers.
    
    Parameters
    ----------
    as_of_date : str, optional
        Date for constituents (not implemented - returns current)
        
    Returns
    -------
    List[str]
        List of ticker symbols
        
    Notes
    -----
    This returns current constituents. For historical constituents,
    consider using survivorship-bias-free data providers.
    """
    try:
        import pandas as pd
        # Wikipedia table of S&P 500 constituents
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        
        # Clean up tickers (replace . with -)
        tickers = [t.replace('.', '-') for t in tickers]
        
        return tickers
        
    except Exception as e:
        warnings.warn(f"Could not fetch S&P 500 constituents: {e}")
        # Fallback to a sample of major tickers
        return [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'BRK-B', 'JPM',
            'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'VZ', 'ADBE',
            'NFLX', 'INTC', 'CMCSA', 'KO', 'PFE', 'T', 'MRK', 'PEP', 'CSCO',
            'XOM', 'ABT', 'CVX', 'WMT', 'TMO', 'CRM', 'ABBV', 'ACN', 'NKE',
            'COST', 'DHR', 'MCD', 'LLY', 'AVGO', 'TXN', 'NEE', 'BMY', 'UNP',
            'PM', 'MDT', 'LIN', 'QCOM', 'HON'
        ]


def download_eurostoxx50_constituents(
    as_of_date: Optional[str] = None,
) -> List[str]:
    """
    Get Euro Stoxx 50 constituent tickers.
    
    Parameters
    ----------
    as_of_date : str, optional
        Date for constituents (not implemented)
        
    Returns
    -------
    List[str]
        List of ticker symbols (Yahoo Finance format)
    """
    # Euro Stoxx 50 constituents (as of 2024)
    # Yahoo Finance uses .PA, .DE, .MC etc. suffixes
    constituents = [
        # Germany (.DE)
        'ADS.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BMW.DE', 'CON.DE',
        'DAI.DE', 'DBK.DE', 'DPW.DE', 'DTE.DE', 'EOAN.DE', 'FRE.DE',
        'HEI.DE', 'IFX.DE', 'LIN.DE', 'MRK.DE', 'MUV2.DE', 'RWE.DE',
        'SAP.DE', 'SIE.DE', 'VOW3.DE',
        # France (.PA)
        'AI.PA', 'AIR.PA', 'BN.PA', 'BNP.PA', 'CA.PA', 'CS.PA',
        'DG.PA', 'EL.PA', 'EN.PA', 'GLE.PA', 'KER.PA', 'LR.PA',
        'MC.PA', 'OR.PA', 'ORA.PA', 'RI.PA', 'SAN.PA', 'SGO.PA',
        'SU.PA', 'TTE.PA', 'VIV.PA',
        # Netherlands (.AS)
        'AD.AS', 'ASML.AS', 'HEIA.AS', 'INGA.AS', 'PHIA.AS', 'PRX.AS',
        # Spain (.MC)
        'BBVA.MC', 'IBE.MC', 'ITX.MC', 'SAN.MC', 'TEF.MC',
        # Italy (.MI)
        'ENEL.MI', 'ENI.MI', 'ISP.MI', 'UCG.MI',
        # Belgium (.BR)
        'ABI.BR', 'KBC.BR',
        # Ireland
        'CRH.L',
        # Finland (.HE)
        'NOKIA.HE',
    ]
    
    return constituents


def get_risk_free_rate(
    start: Optional[str] = None,
    end: Optional[str] = None,
    maturity: str = '3m',
) -> pd.Series:
    """
    Download risk-free rate data from FRED.
    
    Parameters
    ----------
    start : str, optional
        Start date
    end : str, optional
        End date
    maturity : str
        Treasury maturity: '3m', '1y', '10y'
        
    Returns
    -------
    pd.Series
        Daily risk-free rate (annualized)
    """
    try:
        import pandas_datareader.data as web
    except ImportError:
        warnings.warn(
            "pandas-datareader required for FRED data. "
            "Using constant risk-free rate."
        )
        return None
    
    # FRED series codes
    series_codes = {
        '3m': 'DTB3',     # 3-month T-bill
        '1y': 'DTB1YR',   # 1-year T-bill
        '10y': 'DGS10',   # 10-year Treasury
    }
    
    if maturity not in series_codes:
        raise ValueError(f"Unknown maturity: {maturity}")
    
    # Default dates
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    if start is None:
        start = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    try:
        rf = web.DataReader(
            series_codes[maturity],
            'fred',
            start=start,
            end=end
        )
        
        # Convert to decimal (FRED reports as percentage)
        rf = rf.iloc[:, 0] / 100
        
        return rf
        
    except Exception as e:
        warnings.warn(f"Could not download risk-free rate: {e}")
        return None


def download_fama_french_factors(
    start: Optional[str] = None,
    end: Optional[str] = None,
    frequency: str = 'daily',
) -> pd.DataFrame:
    """
    Download Fama-French factor data.
    
    Parameters
    ----------
    start : str, optional
        Start date
    end : str, optional
        End date
    frequency : str
        'daily' or 'monthly'
        
    Returns
    -------
    pd.DataFrame
        Fama-French factors (Mkt-RF, SMB, HML, RF)
    """
    try:
        import pandas_datareader.data as web
    except ImportError:
        raise ImportError(
            "pandas-datareader required. "
            "Install via: pip install pandas-datareader"
        )
    
    dataset = 'F-F_Research_Data_Factors_daily' if frequency == 'daily' else 'F-F_Research_Data_Factors'
    
    try:
        ff = web.DataReader(dataset, 'famafrench', start=start, end=end)
        factors = ff[0]  # First table
        
        # Convert from percentage
        factors = factors / 100
        
        return factors
        
    except Exception as e:
        warnings.warn(f"Could not download Fama-French factors: {e}")
        return None


def validate_price_data(prices: pd.DataFrame) -> dict:
    """
    Validate downloaded price data.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data to validate
        
    Returns
    -------
    dict
        Validation results and warnings
    """
    results = {
        'n_tickers': prices.shape[1],
        'n_dates': prices.shape[0],
        'date_range': (prices.index.min(), prices.index.max()),
        'missing_pct': prices.isna().mean().to_dict(),
        'issues': [],
    }
    
    # Check for excessive missing data
    for col in prices.columns:
        missing_pct = prices[col].isna().mean()
        if missing_pct > 0.1:
            results['issues'].append(f"{col}: {missing_pct:.1%} missing")
    
    # Check for suspicious values
    returns = prices.pct_change()
    for col in prices.columns:
        max_return = returns[col].abs().max()
        if max_return > 0.5:  # 50% daily move
            results['issues'].append(
                f"{col}: suspicious return of {max_return:.1%}"
            )
    
    # Check for gaps
    if len(prices) > 1:
        date_diffs = pd.Series(prices.index).diff().dropna()
        max_gap = date_diffs.max()
        if max_gap > pd.Timedelta(days=7):
            results['issues'].append(f"Large gap in data: {max_gap}")
    
    return results

