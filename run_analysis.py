import argparse
import sys
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, 'src')

from src.data import (fetch_prices, compute_returns, equal_weight_portfolio,
                       get_correlation_matrix, get_summary_stats, validate_data)
from src.risk import (equity_curve, drawdown_series, historical_var, normal_var,
                       cornish_fisher_var, monte_carlo_var, historical_cvar,
                       rolling_var_series, rolling_cvar_series, component_var,
                       marginal_var, scale_var_to_horizon, var_to_dollars,
                       return_statistics)
from src.backtest import (var_exceptions, kupiec_pof_test, christoffersen_independence_test,
                           backtest_multiple_alphas, stress_test_historical,
                           stress_test_hypothetical, exception_clustering_analysis,
                           save_json, save_dataframe)
from src.plot import (plot_equity, plot_drawdown, plot_return_distribution,
                       plot_exceptions_timeline, plot_var_cvar_overlay,
                       plot_qq, plot_rolling_volatility, plot_var_comparison,
                       plot_var_backtest_summary, plot_stress_test_results,
                       plot_component_var)

def main():
    parser = argparse.ArgumentParser(description='Portfolio VaR Analysis')
    parser.add_argument('--tickers', type=str, required=True,
                         help='Comma-separated list of tickers')
    parser.add_argument('--start', type=str, default='2015-01-01',
                         help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                         help='End date (YYYY-MM-DD)')
    parser.add_argument('--alpha', type=float, default=0.95,
                         help='Primary confidence level')
    parser.add_argument('--window', type=int, default=252,
                         help='Rolling window size in days')
    parser.add_argument('--portfolio-value', type=float, default=1000000,
                         help='Portfolio value in dollars')
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")]

    # Data acquisition
    prices = fetch_prices(tickers, args.start, args.end)
    validate_data(prices)
    returns = compute_returns(prices)
    portfolio_returns = equal_weight_portfolio(returns)

    # Summary statistics
    summary_stats = get_summary_stats(returns)
    corr_matrix = get_correlation_matrix(returns)
    summary_stats.to_csv('outputs/summary_statistics.csv')
    corr_matrix.to_csv('outputs/correlation_matrix.csv')

    # VaR calculation
    var_h = historical_var(portfolio_returns, args.alpha)
    var_n = normal_var(portfolio_returns, args.alpha)
    var_cf = cornish_fisher_var(portfolio_returns, args.alpha)
    var_mc, mc_sims = monte_carlo_var(portfolio_returns, args.alpha, n_sims=10000)
    cvar_h = historical_cvar(portfolio_returns, args.alpha)

    var_h_dollars = var_to_dollars(var_h, args.portfolio_value)
    var_10day = scale_var_to_horizon(var_h, 10)
    var_10day_dollars = var_to_dollars(var_10day, args.portfolio_value)

    var_estimates = pd.DataFrame({
        'method': ['Historical', 'Normal', 'Cornish-Fisher', 'Monte Carlo', 'CVaR'],
        'var_pct': [var_h, var_n, var_cf, var_mc, cvar_h],
        'var_dollars': [var_to_dollars(v, args.portfolio_value)
                         for v in [var_h, var_n, var_cf, var_mc, cvar_h]]
    })
    var_estimates.to_csv('outputs/var_estimates.csv', index=False)

    # Rolling VaR and backtesting
    effective_window = min(args.window, max(1, len(portfolio_returns) - 1))

    var_hist_roll = rolling_var_series(portfolio_returns, args.alpha,
                                         effective_window, method='historical')
    var_norm_roll = rolling_var_series(portfolio_returns, args.alpha,
                                         effective_window, method='normal')
    var_cf_roll = rolling_var_series(portfolio_returns, args.alpha,
                                       effective_window, method='cornish')
    cvar_roll = rolling_cvar_series(portfolio_returns, args.alpha, effective_window)

    methods_dict = {
        'Historical': var_hist_roll,
        'Normal': var_norm_roll,
        'Cornish-Fisher': var_cf_roll
    }

    # Multi-confidence level backtesting
    alphas = [0.90, 0.95, 0.975, 0.99]
    var_series_dict = {}
    for alpha in alphas:
        var_series_dict[alpha] = rolling_var_series(
            portfolio_returns, alpha, effective_window, method='historical'
        )

    backtest_results = backtest_multiple_alphas(portfolio_returns, var_series_dict)
    backtest_results.to_csv('outputs/backtest_results.csv', index=False)

    # Stress testing
    historical_scenarios = {
        'Financial Crisis 2008': ('2008-09-01', '2008-12-31'),
        'Flash Crash 2010': ('2010-05-03', '2010-05-10'),
        'EU Debt Crisis 2011': ('2011-08-01', '2011-10-31'),
        'China Devaluation 2015': ('2015-08-17', '2015-09-30'),
        'Brexit 2016': ('2016-06-20', '2016-07-15'),
        'COVID-19 Crash 2020': ('2020-02-19', '2020-03-23'),
        'Russia-Ukraine 2022': ('2022-02-20', '2022-03-15'),
        '2022 Rate Hikes': ('2022-06-01', '2022-10-31')
    }

    stress_results = stress_test_historical(portfolio_returns, historical_scenarios)
    stress_results.to_csv('outputs/stress_test_results.csv', index=False)

    hyp_stress = stress_test_hypothetical(portfolio_returns, var_h)
    hyp_stress.to_csv('outputs/hypothetical_stress.csv', index=False)

    # Component VaR
    if len(tickers) > 1:
        weights = np.array([1/len(tickers)] * len(tickers))
        comp_var = component_var(returns, weights, args.alpha)
        marg_var = marginal_var(returns, weights, args.alpha)

        pd.DataFrame({
            'asset': list(comp_var.keys()),
            'component_var': list(comp_var.values()),
            'marginal_var': list(marg_var.values())
        }).to_csv('outputs/component_var.csv', index=False)

    # Exception clustering analysis
    exc = var_exceptions(portfolio_returns, var_hist_roll)
    clustering = exception_clustering_analysis(exc)

    # Visualizations
    eq = equity_curve(portfolio_returns, initial_value=args.portfolio_value)
    dd = drawdown_series(eq)

    plot_equity(eq, 'visualizations/equity_curve.png')
    plot_drawdown(dd, 'visualizations/drawdown.png')
    plot_return_distribution(portfolio_returns, var_h, var_n, var_cf,
                               'visualizations/return_distribution.png')
    plot_exceptions_timeline(portfolio_returns, var_hist_roll,
                               'visualizations/var_exceptions_timeline.png')
    plot_var_cvar_overlay(portfolio_returns, var_hist_roll, cvar_roll,
                           'visualizations/var_cvar_overlay.png')
    plot_qq(portfolio_returns, 'visualizations/qq_plot.png')
    plot_rolling_volatility(portfolio_returns, [21, 63, 252],
                              'visualizations/rolling_volatility.png')
    plot_var_comparison(portfolio_returns, methods_dict,
                         'visualizations/var_method_comparison.png')
    plot_var_backtest_summary(backtest_results,
                               'visualizations/backtest_summary.png')
    plot_stress_test_results(stress_results,
                               'visualizations/stress_test_results.png')

    if len(tickers) > 1:
        plot_component_var(comp_var, 'visualizations/component_var.png')

    # Summary report
    stats = return_statistics(portfolio_returns)
    summary_report = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tickers': tickers,
        'start_date': args.start,
        'end_date': args.end,
        'n_observations': len(portfolio_returns),
        'portfolio_value': args.portfolio_value,
        'primary_confidence': args.alpha,
        'window_size': effective_window,
        'portfolio_statistics': stats,
        'var_estimates': {
            'historical_pct': float(var_h),
            'normal_pct': float(var_n),
            'cornish_fisher_pct': float(var_cf),
            'monte_carlo_pct': float(var_mc),
            'cvar_pct': float(cvar_h),
            'historical_dollars': float(var_h_dollars),
            'var_10day_dollars': float(var_10day_dollars)
        },
        'backtest_summary': {
            'methods_tested': list(methods_dict.keys()),
            'confidence_levels_tested': alphas
        },
        'clustering_analysis': clustering
    }

    save_json(summary_report, 'outputs/summary_report.json')

if __name__ == "__main__":
    main()
