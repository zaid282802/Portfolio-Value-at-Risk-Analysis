import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from typing import Tuple, Dict

def historical_var(returns: pd.Series, alpha: float) -> float:
    return returns.quantile(1 - alpha)

def historical_cvar(returns: pd.Series, alpha: float) -> float:
    var_level = historical_var(returns, alpha)
    tail = returns[returns <= var_level]
    return tail.mean() if len(tail) > 0 else np.nan

def normal_var(returns: pd.Series, alpha: float) -> float:
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(1 - alpha)
    return mu + z * sigma

def cornish_fisher_var(returns: pd.Series, alpha: float) -> float:
    # Cornish-Fisher expansion for higher moments
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(1 - alpha)
    g1 = skew(returns, bias=False)
    g2 = kurtosis(returns, fisher=True, bias=False)

    z_cf = (z + (1/6)*(z**2 - 1)*g1 + (1/24)*(z**3 - 3*z)*g2
            - (1/36)*(2*z**3 - 5*z)*(g1**2))

    return mu + z_cf * sigma

def monte_carlo_var(returns: pd.Series, alpha: float, n_sims: int = 10000,
                     method: str = 'normal') -> Tuple[float, np.ndarray]:
    if method == 'normal':
        mu, sigma = returns.mean(), returns.std(ddof=1)
        simulated = np.random.normal(mu, sigma, n_sims)
    elif method == 'historical':
        simulated = np.random.choice(returns.values, size=n_sims, replace=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    var = np.percentile(simulated, (1 - alpha) * 100)
    return var, simulated

def component_var(returns: pd.DataFrame, weights: np.ndarray, alpha: float) -> Dict[str, float]:
    # Component VaR decomposition
    port_returns = returns @ weights
    cov_matrix = returns.cov()
    port_vol = np.sqrt(weights @ cov_matrix @ weights)

    marginal_var = (cov_matrix @ weights) / port_vol * norm.ppf(1 - alpha)
    component_vars = weights * marginal_var

    return {asset: cv for asset, cv in zip(returns.columns, component_vars)}

def marginal_var(returns: pd.DataFrame, weights: np.ndarray, alpha: float) -> Dict[str, float]:
    cov_matrix = returns.cov()
    port_vol = np.sqrt(weights @ cov_matrix @ weights)
    marg_contrib = (cov_matrix @ weights) / port_vol
    z = norm.ppf(1 - alpha)
    marginal_vars = marg_contrib * z

    return {asset: mv for asset, mv in zip(returns.columns, marginal_vars)}

def rolling_var_series(returns: pd.Series, alpha: float, window: int,
                        method: str = 'historical') -> pd.Series:
    vals, idxs = [], []

    for i in range(window, len(returns)):
        window_ret = returns.iloc[i-window:i]

        if method == 'historical':
            v = historical_var(window_ret, alpha)
        elif method == 'normal':
            v = normal_var(window_ret, alpha)
        elif method == 'cornish':
            v = cornish_fisher_var(window_ret, alpha)
        elif method == 'monte_carlo':
            v, _ = monte_carlo_var(window_ret, alpha, n_sims=5000)
        else:
            raise ValueError(f"Unknown method: {method}")

        vals.append(v)
        idxs.append(returns.index[i])

    return pd.Series(vals, index=idxs, name=f'VaR_{method}_{int(alpha*100)}')

def rolling_cvar_series(returns: pd.Series, alpha: float, window: int) -> pd.Series:
    vals, idxs = [], []

    for i in range(window, len(returns)):
        window_ret = returns.iloc[i-window:i]
        var_level = window_ret.quantile(1 - alpha)
        tail = window_ret[window_ret <= var_level]
        vals.append(tail.mean() if len(tail) > 0 else np.nan)
        idxs.append(returns.index[i])

    return pd.Series(vals, index=idxs, name=f'CVaR_hist_{int(alpha*100)}')

def equity_curve(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    eq = initial_value * (1 + returns).cumprod()
    eq.name = 'equity'
    return eq

def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    dd.name = 'drawdown'
    return dd

def scale_var_to_horizon(var_1day: float, horizon: int, method: str = 'sqrt') -> float:
    # Square-root-of-time scaling
    if method == 'sqrt':
        return var_1day * np.sqrt(horizon)
    else:
        raise ValueError(f"Unknown scaling method: {method}")

def var_to_dollars(var_pct: float, portfolio_value: float) -> float:
    return var_pct * portfolio_value

def return_statistics(returns: pd.Series) -> Dict[str, float]:
    return {
        'mean': returns.mean(),
        'std': returns.std(ddof=1),
        'skewness': skew(returns, bias=False),
        'kurtosis': kurtosis(returns, fisher=True, bias=False),
        'min': returns.min(),
        'max': returns.max(),
        'median': returns.median(),
        'count': len(returns)
    }
