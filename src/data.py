import pandas as pd
import yfinance as yf
import numpy as np
from typing import List
from scipy.stats import skew, kurtosis

def fetch_prices(tickers: List[str], start: str, end: str, retry_count: int = 3) -> pd.DataFrame:
    tickers = [t.strip().upper() for t in tickers]

    for attempt in range(retry_count):
        try:
            df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

            if isinstance(df, pd.DataFrame):
                if 'Adj Close' in df.columns:
                    prices = df['Adj Close']
                elif 'Close' in df.columns:
                    prices = df['Close']
                else:
                    if df.columns.nlevels > 1:
                        prices = df['Close'] if 'Close' in df.columns.get_level_values(0) else df
                    else:
                        prices = df
            else:
                prices = df

            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0])

            prices = prices.dropna()

            if prices.empty:
                raise ValueError(f"No data retrieved for {tickers}")

            return prices

        except Exception as e:
            if attempt == retry_count - 1:
                raise ValueError(f"Failed to fetch data after {retry_count} attempts: {e}")

    raise ValueError("Failed to fetch data")

def compute_returns(prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
    if method == 'simple':
        returns = prices.pct_change()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown method: {method}")

    returns = returns.dropna()
    returns.index.name = 'date'
    return returns

def equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    n = returns.shape[1]
    weights = pd.Series([1.0/n] * n, index=returns.columns)
    port = returns.dot(weights)
    port.name = 'portfolio'
    return port

def get_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()

def get_summary_stats(returns: pd.DataFrame) -> pd.DataFrame:
    stats = {}
    for col in returns.columns:
        ret = returns[col]
        stats[col] = {
            'Mean (Ann.)': ret.mean() * 252,
            'Volatility (Ann.)': ret.std() * np.sqrt(252),
            'Sharpe (Ann.)': (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else np.nan,
            'Skewness': skew(ret, bias=False),
            'Kurtosis': kurtosis(ret, fisher=True, bias=False),
            'Min': ret.min(),
            'Max': ret.max(),
            'Observations': len(ret)
        }
    return pd.DataFrame(stats).T

def validate_data(prices: pd.DataFrame) -> bool:
    if (prices <= 0).any().any():
        raise ValueError("Data contains zero or negative prices")

    returns = prices.pct_change()
    extreme_returns = (returns.abs() > 0.5).any()
    if extreme_returns.any():
        pass  # Warning only, not fatal

    return True
