# Portfolio VaR Analysis

**October 2025** | Zaid Annigeri | Master of Quantitative Finance, Rutgers Business School

> Comprehensive Value-at-Risk (VaR) framework implementing multiple methodologies, backtesting procedures, and stress testing for multi-asset portfolios.

## Project Overview

This project implements a professional-grade VaR analysis framework covering the complete lifecycle of risk measurement:

- **Multiple VaR Methodologies**: Historical, Parametric (Normal), Cornish-Fisher, and Monte Carlo simulation
- **Rigorous Backtesting**: Kupiec POF test, Christoffersen Independence test, and Basel Traffic Light approach
- **Stress Testing**: Historical crisis scenarios and hypothetical extreme events
- **Component VaR**: Risk decomposition showing each asset's contribution to portfolio risk
- **Regulatory Standards**: 10-day VaR calculation, 99% confidence level testing

**Key Finding**: Historical VaR at 95% confidence failed the Kupiec test (p=0.002), indicating the model underestimates tail risk. This motivated implementation of Cornish-Fisher adjustments for fat-tailed returns and Expected Shortfall (CVaR) as a more conservative risk measure.

---

## Quick Start

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/portfolio-var-analysis.git
cd portfolio-var-analysis

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis

**Full Analysis** (5+ years of data, all VaR methods):
```bash
python run_analysis.py --tickers SPY,QQQ,TLT,GLD --start 2015-01-01 --end 2024-12-31
```

**Custom Portfolio**:
```bash
python run_analysis.py --tickers AAPL,MSFT,GOOGL --start 2018-01-01 --end 2023-12-31 --alpha 0.99 --portfolio-value 5000000
```

**Parameters**:
- `--tickers`: Comma-separated list of tickers (e.g., `SPY,QQQ,TLT,GLD`)
- `--start`: Start date (YYYY-MM-DD format, default: 2015-01-01)
- `--end`: End date (YYYY-MM-DD format, default: 2024-12-31)
- `--alpha`: Confidence level (default: 0.95)
- `--window`: Rolling window size in days (default: 252)
- `--portfolio-value`: Portfolio value in dollars (default: 1,000,000)

**Output**:
- `outputs/`: CSV files with backtest results, stress tests, VaR estimates
- `visualizations/`: 11 professional charts (300 DPI PNG)
- `outputs/summary_report.json`: Complete analysis summary

---

## VaR Methodologies Implemented

### 1. Historical VaR
Non-parametric approach using empirical quantiles of historical return distribution.

**Advantages**:
- No distributional assumptions
- Captures actual tail behavior

**Limitations**:
- Limited by historical sample
- Assumes past represents future

### 2. Parametric VaR (Normal)
Assumes returns follow a normal distribution: VaR = μ + z_α × σ

**Advantages**:
- Simple, fast computation
- Well-understood statistical properties

**Limitations**:
- **Underestimates tail risk** (fat tails in financial returns)
- Ignores skewness and kurtosis

### 3. Cornish-Fisher VaR
Adjusts normal VaR for skewness and excess kurtosis using Cornish-Fisher expansion.

**Formula**:
```
z_CF = z + (1/6)(z² - 1)S + (1/24)(z³ - 3z)K - (1/36)(2z³ - 5z)S²
VaR_CF = μ + z_CF × σ
```

**Advantages**:
- Accounts for non-normality
- More accurate for fat-tailed distributions

**Limitations**:
- Still parametric (assumes expansion converges)

### 4. Monte Carlo VaR
Simulates 10,000+ return paths using fitted distribution or bootstrap resampling.

**Advantages**:
- Flexible (any distribution)
- Provides confidence intervals around VaR

**Limitations**:
- Computationally intensive
- Sensitive to model specification

### 5. Conditional VaR (CVaR) / Expected Shortfall
Average loss beyond VaR threshold: CVaR = E[Return | Return ≤ VaR]

**Advantages**:
- **Coherent risk measure** (satisfies subadditivity)
- Regulatory preference (Basel IV)
- Captures tail severity, not just frequency

---

## Backtesting Framework

### Kupiec Proportion of Failures (POF) Test
Tests whether observed exception rate matches expected rate.

**Null Hypothesis**: Exception rate = (1 - α)

**Test Statistic**: Likelihood ratio (LR) ~ χ²(1)

**Decision Rule**: Reject if p-value < 0.05

### Christoffersen Independence Test
Tests whether exceptions are independently distributed (i.i.d.).

**Why It Matters**: Clustered exceptions indicate model fails to capture volatility dynamics.

### Basel Traffic Light Approach
- **Green Zone**: Model acceptable (exception rate ≤ threshold)
- **Yellow Zone**: Model requires monitoring
- **Red Zone**: Model should be rejected

### Multi-Confidence Level Testing
Tests VaR at 90%, 95%, 97.5%, and 99% to verify model performs consistently across confidence levels.

---

## Stress Testing

### Historical Crisis Scenarios
- **2008 Financial Crisis** (Lehman collapse)
- **2010 Flash Crash**
- **2011 EU Debt Crisis**
- **2015 China Devaluation**
- **2016 Brexit**
- **2020 COVID-19 Crash**
- **2022 Russia-Ukraine War**
- **2022 Fed Rate Hikes**

### Hypothetical Scenarios
- 1σ, 2σ, 3σ market declines
- Black Monday 1987 (-20.5%)
- VaR breach multiples (2x, 3x, 5x)

**Output**: Shows which scenarios would exceed VaR threshold and by how much.

---

## Component VaR Decomposition

For multi-asset portfolios, decomposes total VaR into contributions from each asset:

**Component VaR** = w_i × (∂VaR / ∂w_i)

Shows which positions drive portfolio risk, enabling targeted risk reduction.

**Marginal VaR** = ∂VaR / ∂w_i

Measures how much portfolio VaR would change if we increased position in asset i by 1%.

---

## Project Structure

```
portfolio_var_analysis/
├── run_analysis.py                 # Main analysis script (400+ lines)
├── src/                            # Modular source code
│   ├── data.py                     # Data fetching and preprocessing (250 lines)
│   ├── risk.py                     # VaR calculation functions (400 lines)
│   ├── backtest.py                 # Backtesting framework (450 lines)
│   └── plot.py                     # Visualization functions (700 lines)
├── outputs/                        # Generated analysis results
│   ├── summary_report.json         # Complete analysis summary
│   ├── var_estimates.csv           # VaR estimates for all methods
│   ├── backtest_results.csv        # Backtest results at multiple confidence levels
│   ├── stress_test_results.csv     # Historical stress scenario results
│   ├── component_var.csv           # VaR decomposition by asset
│   └── correlation_matrix.csv      # Asset correlation matrix
├── visualizations/                 # Generated charts (300 DPI PNG)
│   ├── equity_curve.png            # Cumulative portfolio value
│   ├── drawdown.png                # Drawdown over time
│   ├── return_distribution.png     # Histogram with VaR thresholds
│   ├── var_exceptions_timeline.png # VaR breaches over time
│   ├── var_cvar_overlay.png        # Rolling VaR and CVaR
│   ├── qq_plot.png                 # Q-Q plot (normality test)
│   ├── rolling_volatility.png      # 21/63/252-day rolling volatility
│   ├── var_method_comparison.png   # All VaR methods on one chart
│   ├── backtest_summary.png        # 4-panel backtest visualization
│   ├── stress_test_results.png     # Stress scenario returns
│   └── component_var.png           # VaR contribution pie chart
├── report/                         # LaTeX report and PDF
│   └── VaR_Report.pdf              # Compiled PDF report
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```
---

## Key Visualizations

### 1. Equity Curve
Shows cumulative portfolio value over time.

### 2. Drawdown Analysis
Identifies maximum drawdown and recovery periods.

### 3. Return Distribution with VaR
Histogram overlaid with:
- Fitted normal distribution (shows fat tails)
- Historical, Normal, Cornish-Fisher VaR thresholds

### 4. VaR Exceptions Timeline
2-panel chart:
- Top: Returns with VaR threshold and exception markers
- Bottom: Binary exception indicators

### 5. Q-Q Plot
Tests normality assumption by comparing empirical quantiles to theoretical normal quantiles.

**Key Insight**: Deviations in tails indicate fat-tailed distribution, motivating Cornish-Fisher adjustment.

### 6. Rolling Volatility
Shows 21-day, 63-day, and 252-day rolling volatility to identify regime changes.

### 7. VaR Method Comparison
Overlays Historical, Normal, and Cornish-Fisher VaR on same chart.

**Key Insight**: Normal VaR consistently less conservative than Historical/Cornish-Fisher.

### 8. Backtest Summary (4-panel)
- Exception counts (actual vs expected)
- Kupiec p-values (green = pass, red = fail)
- Traffic light status
- Exception rate comparison

### 9. Stress Test Results
Bar charts showing portfolio loss during historical crises.

### 10. Component VaR
Pie chart showing % of total VaR contributed by each asset.

---

## Why This VaR Model Failed (And What We Learned)

### The Problem
```json
{
  "kupiec_test": {
    "p_value": 0.002,    # REJECTED at 1% significance
    "exceptions": 15,
    "n": 125,
    "obs_rate": 0.12,    # 12% vs expected 5%
    "decision": "reject"
  }
}
```

### Root Causes

1. **Fat-Tailed Returns**
   - Financial returns exhibit excess kurtosis
   - Normal distribution underestimates probability of extreme events

2. **Volatility Clustering**
   - GARCH effects: large moves followed by large moves
   - VaR model assumes i.i.d. returns, which fails during market stress

3. **Regime Changes**
   - 2018 Q4 volatility spike (VIX hit 36)
   - Model estimated on calm period, tested on volatile period

### Solutions Implemented

1. **Cornish-Fisher Adjustment**
   - Accounts for skewness (-0.45) and kurtosis (8.2)
   - VaR_CF = -1.83% vs VaR_Normal = -1.78%

2. **Expected Shortfall (CVaR)**
   - Measures average loss beyond VaR
   - More conservative: CVaR = -2.51% vs VaR = -2.09%

3. **Dynamic Window**
   - Shorter windows during volatile periods
   - EWMA weighting for recent observations

4. **Multiple Confidence Levels**
   - Test at 90%, 95%, 97.5%, 99%
   - Ensures model robust across risk thresholds

---

## Practical Interpretation

### For a $1,000,000 Portfolio

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **1-Day VaR (95%)** | $20,900 | 5% chance of losing more than $20,900 tomorrow |
| **10-Day VaR (99%)** | $66,100 | 1% chance of losing more than $66,100 over 10 days (Basel regulatory standard) |
| **CVaR (95%)** | $25,100 | If we breach VaR, expected loss is $25,100 |

### Compared to Industry Benchmarks
- **Typical Equity Fund VaR**: 1.5-2.0%
- **Our Portfolio VaR**: 2.09% (slightly above average)
- **Interpretation**: Higher risk than typical diversified equity fund

---

## Regulatory Context

### Basel III Market Risk Framework
- **VaR Standard**: 99% confidence, 10-day horizon
- **Stressed VaR**: VaR calculated on 1-year stressed period
- **Capital Requirement**: max(VaR, SVaR) × 3 × multiplier

### Basel IV (Fundamental Review of Trading Book)
- **Shift to Expected Shortfall**: More sensitive to tail risk
- **VaR Limitations Recognized**:
  - Not coherent (fails subadditivity)
  - Doesn't capture tail severity
  - Prone to gaming

---

## Limitations

1. **Sample Period Dependency**
   - Results sensitive to chosen start/end dates
   - Bull market (2010-2020) vs bear market yields different VaR

2. **Assumes Liquid Markets**
   - VaR doesn't account for liquidation costs
   - Position sizes may exceed market capacity

3. **No Tail Correlation**
   - Assumes asset correlations stable across regimes
   - Reality: Correlations → 1 during crises

4. **Static Portfolio**
   - Assumes no trading or rebalancing
   - Real portfolios dynamic

5. **Model Risk**
   - All models are wrong, some are useful
   - VaR is not worst-case scenario

---

## Technology Stack

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical computations, matrix operations |
| `pandas` | Time series data manipulation |
| `yfinance` | Historical price data from Yahoo Finance |
| `scipy` | Statistical tests (Kupiec, Christoffersen), distributions |
| `matplotlib` | Base plotting library |
| `seaborn` | Statistical visualization, styling |

---

## Sample Output

```
================================================================================
PORTFOLIO VAR ANALYSIS
================================================================================

95.0% VaR Estimates:
  Historical:       -0.0209 (-2.09%)
  Normal:           -0.0178 (-1.78%)
  Cornish-Fisher:   -0.0183 (-1.83%)
  Monte Carlo:      -0.0207 (-2.07%)
  CVaR (Historical): -0.0251 (-2.51%)

1-Day VaR ($1,000,000 portfolio):
  Historical VaR: $20,900
  10-Day VaR (√10 scaling): $66,100

Backtest Results (95% Confidence):

Historical:
  Exceptions: 15 / 125
  Exception Rate: 12.00% (expected: 5.00%)
  Kupiec p-value: 0.0021 (reject)

Cornish-Fisher:
  Exceptions: 8 / 125
  Exception Rate: 6.40% (expected: 5.00%)
  Kupiec p-value: 0.4521 (accept)
```

---

## Author

**Zaid Annigeri**
- Master of Quantitative Finance, Rutgers Business School
- GitHub: [@Zaid282802](https://github.com/Zaid282802)
- LinkedIn: [Zaid Annigeri](https://www.linkedin.com/in/zed228)
- Email: mz845@scarletmail.rutgers.edu
---

## License

This project is available for educational and research purposes. Please provide attribution if used in academic work or presentations.

**Citation:**
```
[Zaid Annigeri] (2025). Portfolio VaR Analysis: Comprehensive Risk Measurement Framework.
Master of Quantitative Finance Program, Rutgers Business School.
```

---

## References

1. Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*. McGraw-Hill.

2. Kupiec, P. (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models". *Journal of Derivatives*, 3(2), 73-84.

3. Christoffersen, P. (1998). "Evaluating Interval Forecasts". *International Economic Review*, 39(4), 841-862.

4. Favre, L., & Galeano, J. A. (2002). "Mean-Modified Value-at-Risk Optimization with Hedge Funds". *Journal of Alternative Investments*, 5(2), 21-25.

5. Basel Committee on Banking Supervision (2019). *Minimum Capital Requirements for Market Risk*.

