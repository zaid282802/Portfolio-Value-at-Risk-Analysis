import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

def plot_equity(equity: pd.Series, output_path: str, benchmark: Optional[pd.Series] = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity.index, equity.values, label='Portfolio', linewidth=2, color='#2E86AB')

    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, label='Benchmark',
                 linewidth=2, linestyle='--', color='#A23B72', alpha=0.7)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Value ($)', fontsize=12)
    ax.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_drawdown(drawdown: pd.Series, output_path: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(drawdown.index, drawdown.values, 0, color='#C1121F', alpha=0.7, label='Drawdown')
    ax.plot(drawdown.index, drawdown.values, color='#780000', linewidth=1.5)

    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    ax.scatter([max_dd_idx], [max_dd_val], color='red', s=100, zorder=5,
                label=f'Max DD: {max_dd_val:.2%}')
    ax.annotate(f'{max_dd_val:.2%}', xy=(max_dd_idx, max_dd_val),
                 xytext=(10, -10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_return_distribution(returns: pd.Series, var_h: float, var_n: float,
                               var_cf: float, output_path: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(returns, bins=50, alpha=0.7, color='#4A90E2', edgecolor='black',
             density=True, label='Returns')

    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
             label='Normal Distribution', alpha=0.7)

    ax.axvline(var_h, color='green', linestyle='--', linewidth=2,
                label=f'Historical VaR: {var_h:.4f}')
    ax.axvline(var_n, color='blue', linestyle='--', linewidth=2,
                label=f'Normal VaR: {var_n:.4f}')
    ax.axvline(var_cf, color='orange', linestyle='--', linewidth=2,
                label=f'Cornish-Fisher VaR: {var_cf:.4f}')

    ax.set_xlabel('Daily Returns', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Return Distribution with VaR Estimates (95% Confidence)',
                  fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_exceptions_timeline(returns: pd.Series, var_series: pd.Series, output_path: str):
    aligned_ret = returns.loc[var_series.index]
    exceptions = aligned_ret <= var_series

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(aligned_ret.index, aligned_ret.values, label='Returns',
              linewidth=1, color='gray', alpha=0.7)
    ax1.plot(var_series.index, var_series.values, label='VaR Threshold',
              linewidth=2, color='red', linestyle='--')

    exc_dates = aligned_ret[exceptions].index
    exc_values = aligned_ret[exceptions].values
    ax1.scatter(exc_dates, exc_values, color='red', s=50, zorder=5,
                 label=f'Exceptions (n={exceptions.sum()})')

    ax1.set_ylabel('Daily Returns', fontsize=12)
    ax1.set_title('VaR Exceptions Timeline', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linewidth=0.8)

    exc_binary = exceptions.astype(int)
    ax2.fill_between(exc_binary.index, 0, exc_binary.values,
                       color='red', alpha=0.5, step='mid')
    ax2.set_ylabel('Exception', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['No', 'Yes'])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_var_cvar_overlay(returns: pd.Series, var_series: pd.Series,
                            cvar_series: pd.Series, output_path: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    aligned_ret = returns.loc[var_series.index]

    ax.plot(aligned_ret.index, aligned_ret.values, label='Returns',
             linewidth=1, color='gray', alpha=0.5)
    ax.plot(var_series.index, var_series.values, label='VaR (95%)',
             linewidth=2, color='#E74C3C')
    ax.plot(cvar_series.index, cvar_series.values, label='CVaR (95%)',
             linewidth=2, color='#8E44AD', linestyle='--')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Returns / Risk Metrics', fontsize=12)
    ax.set_title('Rolling VaR and CVaR', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_qq(returns: pd.Series, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 8))
    stats.probplot(returns, dist="norm", plot=ax)

    ax.set_title('Q-Q Plot: Returns vs Normal Distribution',
                  fontsize=14, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles (Normal)', fontsize=12)
    ax.set_ylabel('Sample Quantiles (Actual Returns)', fontsize=12)
    ax.grid(True, alpha=0.3)

    from scipy.stats import normaltest
    stat, p_value = normaltest(returns)
    textstr = f'Normality Test\nStatistic: {stat:.2f}\np-value: {p_value:.4f}'
    if p_value < 0.05:
        textstr += '\nReject normality (p<0.05)'
    else:
        textstr += '\nFail to reject normality'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_rolling_volatility(returns: pd.Series, windows: List[int], output_path: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    for window, color in zip(windows, colors):
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        ax.plot(rolling_vol.index, rolling_vol.values,
                 label=f'{window}-day (Ann.)', linewidth=2, color=color)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Annualized Volatility', fontsize=12)
    ax.set_title('Rolling Volatility Analysis', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_var_comparison(returns: pd.Series, var_methods: Dict[str, pd.Series], output_path: str):
    fig, ax = plt.subplots(figsize=(14, 7))
    common_idx = var_methods[list(var_methods.keys())[0]].index
    aligned_ret = returns.loc[common_idx]
    ax.plot(aligned_ret.index, aligned_ret.values, label='Returns',
             linewidth=1, color='gray', alpha=0.3)

    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    for (method, var_series), color in zip(var_methods.items(), colors):
        ax.plot(var_series.index, var_series.values, label=method,
                 linewidth=2.5, color=color, alpha=0.8)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Returns / VaR', fontsize=12)
    ax.set_title('VaR Method Comparison (95% Confidence)',
                  fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_var_backtest_summary(backtest_df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x_pos = np.arange(len(backtest_df))

    # Panel 1: Exception rates
    ax = axes[0, 0]
    ax.bar(x_pos, backtest_df['exceptions'], alpha=0.7, color='#E74C3C', label='Actual Exceptions')
    expected = backtest_df.apply(lambda row: (1 - row['alpha']) * row['n_obs'], axis=1)
    ax.plot(x_pos, expected, 'o-', color='#2ECC71', linewidth=2,
             markersize=8, label='Expected Exceptions')
    ax.set_xlabel('Confidence Level', fontsize=11)
    ax.set_ylabel('Number of Exceptions', fontsize=11)
    ax.set_title('VaR Exceptions: Actual vs Expected', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(backtest_df['confidence'], rotation=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Kupiec p-values
    ax = axes[0, 1]
    colors = ['green' if p > 0.05 else 'red' for p in backtest_df['kupiec_pval']]
    ax.bar(x_pos, backtest_df['kupiec_pval'], alpha=0.7, color=colors)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='5% Significance Level')
    ax.set_xlabel('Confidence Level', fontsize=11)
    ax.set_ylabel('Kupiec Test p-value', fontsize=11)
    ax.set_title('Kupiec POF Test Results', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(backtest_df['confidence'], rotation=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Traffic light
    ax = axes[1, 0]
    traffic_colors = {'green': '#2ECC71', 'yellow': '#F39C12', 'red': '#E74C3C'}
    colors = [traffic_colors[t] for t in backtest_df['traffic_light']]
    ax.bar(x_pos, [1]*len(backtest_df), alpha=0.7, color=colors)
    ax.set_xlabel('Confidence Level', fontsize=11)
    ax.set_ylabel('Status', fontsize=11)
    ax.set_title('Traffic Light Test (Basel Approach)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(backtest_df['confidence'], rotation=0)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ECC71', label='Green (Accept)'),
        Patch(facecolor='#F39C12', label='Yellow (Monitor)'),
        Patch(facecolor='#E74C3C', label='Red (Reject)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Panel 4: Exception rates comparison
    ax = axes[1, 1]
    obs_rates = [float(r.strip('%')) for r in backtest_df['observed_rate']]
    exp_rates = [float(r.strip('%')) for r in backtest_df['expected_rate']]
    x = np.arange(len(backtest_df))
    width = 0.35
    ax.bar(x - width/2, exp_rates, width, label='Expected', alpha=0.7, color='#3498DB')
    ax.bar(x + width/2, obs_rates, width, label='Observed', alpha=0.7, color='#E74C3C')
    ax.set_xlabel('Confidence Level', fontsize=11)
    ax.set_ylabel('Exception Rate (%)', fontsize=11)
    ax.set_title('Exception Rate Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(backtest_df['confidence'], rotation=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_stress_test_results(stress_df: pd.DataFrame, output_path: str):
    stress_df = stress_df[stress_df['status'] == 'completed'].copy()

    if len(stress_df) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scenarios = stress_df['scenario']
    total_returns = stress_df['total_return'] * 100
    colors = ['red' if r < 0 else 'green' for r in total_returns]

    ax1.barh(scenarios, total_returns, color=colors, alpha=0.7)
    ax1.set_xlabel('Total Return (%)', fontsize=12)
    ax1.set_title('Stress Test: Scenario Returns', fontsize=13, fontweight='bold')
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.grid(True, alpha=0.3, axis='x')

    worst_days = stress_df['worst_day'] * 100
    colors = ['darkred' if w < -5 else 'orange' for w in worst_days]

    ax2.barh(scenarios, worst_days, color=colors, alpha=0.7)
    ax2.set_xlabel('Worst Single Day (%)', fontsize=12)
    ax2.set_title('Stress Test: Worst Day Loss', fontsize=13, fontweight='bold')
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_component_var(component_var_dict: Dict[str, float], output_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    assets = list(component_var_dict.keys())
    contributions = list(component_var_dict.values())

    total = sum([abs(c) for c in contributions])
    pct_contributions = [abs(c) / total * 100 for c in contributions]

    colors = plt.cm.Set3(np.linspace(0, 1, len(assets)))
    ax.pie(pct_contributions, labels=assets, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax.set_title('Component VaR Decomposition', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
