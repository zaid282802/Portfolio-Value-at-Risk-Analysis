import json
import numpy as np
import pandas as pd
from scipy.stats import chi2
from typing import Dict, List, Tuple

def var_exceptions(returns: pd.Series, var_series: pd.Series) -> pd.Series:
    aligned_ret = returns.loc[var_series.index]
    exc = (aligned_ret <= var_series).astype(int)
    exc.name = 'exceptions'
    return exc

def kupiec_pof_test(exceptions: pd.Series, alpha: float) -> dict:
    # Kupiec Proportion of Failures test
    n = len(exceptions)
    x = exceptions.sum()
    p = 1 - alpha

    if n == 0 or x == 0 or x == n:
        return {
            "LR": np.nan, "p_value": np.nan, "exceptions": int(x), "n": int(n),
            "obs_rate": x/n if n > 0 else np.nan, "expected_rate": p,
            "decision": "insufficient_data"
        }

    lr = -2 * np.log(((1-p)**(n-x) * (p**x)) / (((1 - x/n)**(n-x)) * ((x/n)**x)))
    p_value = 1 - chi2.cdf(lr, df=1)
    decision = "reject" if p_value < 0.05 else "accept"

    return {
        "LR": float(lr), "p_value": float(p_value), "exceptions": int(x),
        "n": int(n), "obs_rate": float(x/n), "expected_rate": float(p),
        "decision": decision
    }

def christoffersen_independence_test(exceptions: pd.Series) -> dict:
    # Christoffersen Independence test
    exc = exceptions.values
    n00 = n01 = n10 = n11 = 0

    for i in range(1, len(exc)):
        if exc[i-1] == 0 and exc[i] == 0: n00 += 1
        elif exc[i-1] == 0 and exc[i] == 1: n01 += 1
        elif exc[i-1] == 1 and exc[i] == 0: n10 += 1
        elif exc[i-1] == 1 and exc[i] == 1: n11 += 1

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    L0 = ((1-pi)**(n00+n10)) * (pi**(n01+n11))
    L1 = ((1-pi0)**n00) * (pi0**n01) * ((1-pi1)**n10) * (pi1**n11)

    LR = -2 * np.log(L0/L1) if L0 > 0 and L1 > 0 else np.nan
    p_value = 1 - chi2.cdf(LR, df=1) if LR == LR else np.nan
    decision = "reject" if p_value < 0.05 else "accept" if p_value == p_value else "insufficient_data"

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi0": float(pi0), "pi1": float(pi1),
        "LR": float(LR) if LR == LR else None,
        "p_value": float(p_value) if p_value == p_value else None,
        "decision": decision
    }

def traffic_light_test(exceptions: int, n: int, alpha: float) -> str:
    # Basel traffic light approach
    expected = (1 - alpha) * n
    obs_rate = exceptions / n if n > 0 else 0

    if alpha >= 0.99:
        green_threshold = 4 / 250
        yellow_threshold = 9 / 250
    else:
        expected_rate = 1 - alpha
        green_threshold = expected_rate * 1.5
        yellow_threshold = expected_rate * 3.0

    if obs_rate <= green_threshold:
        return 'green'
    elif obs_rate <= yellow_threshold:
        return 'yellow'
    else:
        return 'red'

def backtest_multiple_alphas(returns: pd.Series, var_series_dict: Dict[float, pd.Series]) -> pd.DataFrame:
    results = []

    for alpha, var_series in var_series_dict.items():
        exc = var_exceptions(returns, var_series)
        kupiec = kupiec_pof_test(exc, alpha)
        christ = christoffersen_independence_test(exc)
        traffic = traffic_light_test(kupiec['exceptions'], kupiec['n'], alpha)

        results.append({
            'confidence': f"{alpha*100:.1f}%",
            'alpha': alpha,
            'var_method': var_series.name,
            'expected_rate': f"{(1-alpha)*100:.1f}%",
            'observed_rate': f"{kupiec['obs_rate']*100:.1f}%",
            'exceptions': kupiec['exceptions'],
            'n_obs': kupiec['n'],
            'kupiec_LR': kupiec['LR'],
            'kupiec_pval': kupiec['p_value'],
            'kupiec_decision': kupiec['decision'],
            'christ_LR': christ['LR'],
            'christ_pval': christ['p_value'],
            'christ_decision': christ['decision'],
            'traffic_light': traffic
        })

    return pd.DataFrame(results)

def stress_test_historical(returns: pd.Series, scenarios: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    results = []

    for scenario_name, (start, end) in scenarios.items():
        try:
            scenario_returns = returns.loc[start:end]

            if len(scenario_returns) == 0:
                results.append({
                    'scenario': scenario_name, 'start_date': start, 'end_date': end,
                    'total_return': np.nan, 'worst_day': np.nan,
                    'volatility': np.nan, 'n_days': 0, 'status': 'no_data'
                })
                continue

            total_return = (1 + scenario_returns).prod() - 1
            worst_day = scenario_returns.min()
            volatility = scenario_returns.std() * np.sqrt(252)
            n_days = len(scenario_returns)

            results.append({
                'scenario': scenario_name, 'start_date': start, 'end_date': end,
                'total_return': total_return, 'worst_day': worst_day,
                'volatility': volatility, 'n_days': n_days, 'status': 'completed'
            })

        except Exception as e:
            results.append({
                'scenario': scenario_name, 'start_date': start, 'end_date': end,
                'total_return': np.nan, 'worst_day': np.nan,
                'volatility': np.nan, 'n_days': 0, 'status': f'error: {str(e)}'
            })

    return pd.DataFrame(results)

def stress_test_hypothetical(returns: pd.Series, var_95: float) -> pd.DataFrame:
    mu = returns.mean()
    sigma = returns.std()

    scenarios = {
        '1-sigma decline': mu - 1*sigma,
        '2-sigma decline': mu - 2*sigma,
        '3-sigma decline': mu - 3*sigma,
        'Black Monday (1987)': -0.2054,
        'Flash Crash (2010)': -0.0498,
        '2x VaR breach': 2 * var_95,
        '3x VaR breach': 3 * var_95,
        '5x VaR breach': 5 * var_95
    }

    results = []
    for name, loss in scenarios.items():
        var_multiple = loss / var_95 if var_95 != 0 else np.nan
        results.append({
            'scenario': name,
            'hypothetical_loss': loss,
            'var_95': var_95,
            'var_multiple': var_multiple,
            'exceeds_var': 'Yes' if loss < var_95 else 'No'
        })

    return pd.DataFrame(results)

def exception_clustering_analysis(exceptions: pd.Series) -> Dict[str, float]:
    exc_indices = exceptions[exceptions == 1].index

    if len(exc_indices) < 2:
        return {
            'total_exceptions': int(exceptions.sum()),
            'avg_time_between': np.nan,
            'max_gap': np.nan,
            'min_gap': np.nan,
            'clustering_ratio': np.nan
        }

    gaps = [(exc_indices[i+1] - exc_indices[i]).days for i in range(len(exc_indices) - 1)]
    expected_gap = len(exceptions) / exceptions.sum() if exceptions.sum() > 0 else np.nan
    avg_gap = np.mean(gaps) if gaps else np.nan
    clustering_ratio = avg_gap / expected_gap if expected_gap > 0 else np.nan

    return {
        'total_exceptions': int(exceptions.sum()),
        'avg_time_between': avg_gap,
        'max_gap': max(gaps) if gaps else np.nan,
        'min_gap': min(gaps) if gaps else np.nan,
        'clustering_ratio': float(clustering_ratio) if clustering_ratio == clustering_ratio else np.nan
    }

def save_json(report: dict, path: str):
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)

def save_dataframe(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
