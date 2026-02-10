# Financial Time Series Analysis: Elbit Systems Stock Returns

A rigorous empirical study of logarithmic stock returns properties for Value at Risk (VaR) modeling, examining eight stylized facts of financial time series using advanced econometric techniques in R.

## 📊 Project Overview

This project conducts a comprehensive statistical analysis of Elbit Systems (ESLT) stock returns from 2015-2024, systematically testing eight fundamental properties of financial time series that are critical for accurate Value at Risk modeling and risk management.

**Academic Context**: Value at Risk Project  
**Author**: Mohamed Anas Baji  
**Date**: October 16, 2024  
**Analysis Period**: January 1, 2015 - September 16, 2024

## 🎯 Objectives

1. **Validate stylized facts** of financial returns through rigorous statistical testing
2. **Model volatility dynamics** using ARMA-GARCH specifications
3. **Test stationarity** with multiple unit root tests (Dickey-Fuller, ADF, Zivot-Andrews, Lee-Strazicich)
4. **Detect structural breaks** in time series using endogenous break-point tests
5. **Prepare foundation** for Value at Risk estimation and backtesting

## 📈 Company Background: Elbit Systems (ESLT)

**Elbit Systems** is an Israeli defense contractor founded in 1967, headquartered in Haifa.
- **Global Ranking**: 21st worldwide in armament production (2023)
- **Primary Supplier**: Ground vehicles and drones for the Israeli Defense Forces
- **Exchange**: Listed on NASDAQ
- **Sector**: Defense & Aerospace

## 🔬 Eight Stylized Facts Analyzed

### Property 1: Asymmetric Loss/Gain Distribution
**Test**: D'Agostino skewness test  
**Finding**: Negative skewness (-0.239) → Rare but severe losses, frequent small gains  
**Implication**: Fat left tail critical for VaR estimation

### Property 2: Fat-Tailed Distribution (Leptokurtosis)
**Test**: Anscombe-Glynn kurtosis test  
**Finding**: Kurtosis = 6.46 (significantly > 3)  
**Implication**: Extreme events more frequent than normal distribution predicts

### Property 3: Autocorrelation Structure
**Tests**: Ljung-Box statistic, ACF/PACF analysis, EACF  
**Findings**:
- Returns: Weak autocorrelation (some at lags 6, 13, 31; strong at lag 9)
- Squared returns: Strong autocorrelation → evidence of volatility clustering
**Model**: ARMA(5,9) → simplified to MA(6,9) with only significant coefficients

### Property 4: Volatility Clustering
**Test**: ARCH-LM test (Engle, 1982)  
**Finding**: Significant ARCH effects at lags 1, 2, 20, 40  
**Interpretation**: "Large changes tend to be followed by large changes" - Mandelbrot

### Property 5: Conditional Fat Tails
**Model**: GARCH(1,1) on standardized ARMA residuals  
**Test**: Kurtosis of GARCH residuals  
**Finding**: Kurtosis = 5.47 (even after GARCH modeling!)  
**Implication**: Need for heavier-tailed distributions (e.g., Student-t)

### Property 6: Leverage Effect
**Method**: Recursive 22-day rolling volatility vs. log prices  
**Finding**: **Present in training period** (rte) → Market drops increase volatility more than equivalent rises  
**Finding**: **Absent in test period** (rtt) → No asymmetric volatility response

### Property 7: Seasonality
**Tests**:
- **Week Effect**: Volatility highest on Mondays, declining through Friday (French & Roll, 1986)
- **January Effect**: Not detected - no abnormal returns in January

### Property 8: Stationarity
**Tests Applied**:
1. **Dickey-Fuller (DF)**: Specification selection (trend/drift/none)
2. **Augmented Dickey-Fuller (ADF)**: Accounts for autocorrelation
3. **Zivot-Andrews (ZA)**: Endogenous single structural break (detected: December 10, 2020)
4. **Lee-Strazicich (LS)**: Endogenous dual breaks (break detected: May 10, 2018)

**Conclusion**: Series is **stationary** (no unit root), with structural breaks during major market events

## 🛠️ Methodology

### Data Preparation
```r
# Train-test split
rte <- returns[2015-01-01:2020-12-31]  # Training: 66% (1510 obs)
rtt <- returns[2021-01-01:2024-09-16]  # Testing: 33% (930 obs)
```

### Statistical Tests Framework

| Property | Statistical Test | Null Hypothesis | Decision Rule |
|----------|-----------------|-----------------|---------------|
| 1. Asymmetry | D'Agostino | Skewness = 0 | Reject if p < 0.05 |
| 2. Fat Tails | Anscombe-Glynn | Kurtosis = 3 | Reject if p < 0.05 |
| 3. Autocorrelation | Ljung-Box | ρ(k) = 0 ∀k | Reject if p < 0.05 |
| 4. ARCH Effects | ARCH-LM | α₁=...=αₘ=0 | Reject if p < 0.05 |
| 5. Cond. Tails | Anscombe on GARCH residuals | Kurt = 3 | Reject if p < 0.05 |
| 6. Leverage | Visual + correlation | - | Graphical inspection |
| 7. Seasonality | Day-of-week analysis | - | Variance comparison |
| 8. Stationarity | DF/ADF/ZA/LS | Unit root exists | Reject if t-stat < critical |

### Model Specification

#### Step 1: Mean Equation (ARMA)
Starting point: EACF suggests ARMA(5,9)  
Final model: Iterative coefficient elimination → MA(6,9) with 2 significant terms

```r
ARMA model: rt = μ + θ₆εₜ₋₆ + θ₉εₜ₋₉ + εₜ
```

#### Step 2: Volatility Equation (GARCH)
```r
σₜ² = α₀ + α₁εₜ₋₁² + β₁σₜ₋₁²
```

**Estimated GARCH(1,1)**:
- α₀ = 0.0447
- α₁ = 0.0944 (ARCH parameter)
- β₁ = 0.8636 (GARCH parameter)
- Persistence: α₁ + β₁ = 0.9580 (high!)

### Unit Root Testing Procedure

#### 1. Dickey-Fuller Test
```
Specification selection:
- Trend model: Test β₁ (trend coefficient)
- Drift model: Test β₀ (intercept)
- None model: Test ρ (AR coefficient)
```

#### 2. Augmented Dickey-Fuller
```r
# Schwert (1989) maximum lag selection
Pmax = floor(12 × (T/100)^0.25)

# Ng-Perron (2001) MAIC criterion
MAIC(p) = ln(σ̂ₚ²) + 2(τₜ(p) + p)/(T - Pmax)
```

#### 3. Zivot-Andrews (Single Break)
**Models tested**:
- "crash": Break in level only
- "both": Break in level and trend

**Break detection**: Endogenous (data-driven)  
**Training period break**: December 10, 2020  
**Test period break**: March 29, 2022

#### 4. Lee-Strazicich (Dual Breaks)
**Advantage**: Breaks allowed under both H₀ and H₁ (more robust)  
**Result**: Confirms stationarity with structural instability

## 📊 Key Results Summary

### Training Period (2015-2020) - rte

| Property | Present? | Key Finding |
|----------|----------|-------------|
| 1. Asymmetry | ✓ | Skew = -0.239 (p < 0.001) |
| 2. Fat Tails | ✓ | Kurt = 6.46 (p < 0.001) |
| 3. Autocorrelation | ✓ | MA(6,9) model |
| 4. Vol. Clustering | ✓ | ARCH effects at all lags |
| 5. Conditional Fat Tails | ✓ | GARCH residuals kurt = 5.47 |
| 6. Leverage Effect | ✓ | Asymmetric volatility response |
| 7. Seasonality | ✓ | Monday effect confirmed |
| 8. Stationarity | ✓ | With break Dec 2020 |

### Test Period (2021-2024) - rtt

| Property | Present? | Key Finding |
|----------|----------|-------------|
| 1. Asymmetry | ✓ | Positive skew = 0.020 (symmetric) |
| 2. Fat Tails | ✓ | Kurt = 10.89 (even higher!) |
| 3. Autocorrelation | ✓ | MA(20) model |
| 4. Vol. Clustering | ✓ | ARCH effects at lags 20, 40 |
| 5. Conditional Fat Tails | ✓ | GARCH residuals kurt = 12.21 |
| 6. Leverage Effect | ✗ | No asymmetric response |
| 7. Seasonality | ✓ | Monday effect confirmed |
| 8. Stationarity | ✓ | With break March 2022 |

## 🛠️ Technologies & Libraries

### R Packages
```r
# Time Series Analysis
library(forecast)      # ARMA/ARIMA modeling
library(TSA)          # Time series diagnostics
library(urca)         # Unit root tests (DF, ADF)

# Financial Time Series
library(yfR)          # Yahoo Finance data retrieval
library(FinTS)        # ARCH testing
library(tseries)      # GARCH modeling

# Statistical Testing
library(moments)      # Skewness & kurtosis tests
library(lmtest)       # Coefficient significance testing
library(CADFtest)     # Covariate-augmented ADF

# Structural Breaks
library(strucchange)  # (implicit in urca for ZA, LS tests)
```

## 📁 Project Structure

### Main Analysis (Rmd)
```
I. Data Presentation
   - Company background
   - Data retrieval and preprocessing
   - Train-test split

II. Eight Properties Analysis
   1. Asymmetry testing
   2. Fat tail testing
   3. Autocorrelation modeling
   4. ARCH effect testing
   5. Conditional fat tails
   6. Leverage effect
   7. Seasonality analysis
   8. Stationarity testing
      - Dickey-Fuller
      - Augmented DF
      - Zivot-Andrews
      - Lee-Strazicich

III. Conclusion
   - Properties summary table
   - Implications for VaR
```

### Outputs
- **PDF Report**: Full analysis with 49 pages, mathematical formulas, tables, and visualizations
- **Figures**: Time series plots, correlograms, volatility graphs, unit root test plots

## 💡 Key Insights for Risk Management

### For VaR Estimation:
1. **Normal distribution inadequate**: Kurtosis of 6.46-10.89 requires Student-t or EVT
2. **Volatility clustering**: Dynamic VaR models (GARCH-based) essential
3. **Asymmetric risk**: Left tail heavier → focus on downside risk
4. **Structural breaks**: Rolling window or regime-switching models needed
5. **High persistence**: GARCH persistence (0.96) → shocks decay slowly

### For Backtesting:
- **Stationarity confirmed** → Standard backtesting procedures valid
- **Structural breaks present** → Consider sub-period analysis
- **Leverage effect** (training) → Conditional coverage tests important
- **No leverage** (test) → Different risk dynamics in recent period

## 📈 Practical Applications

### 1. Value at Risk Estimation
```r
# Next steps with this analysis:
# - Estimate parametric VaR using GARCH(1,1)
# - Estimate historical VaR accounting for fat tails
# - Estimate EVT-based VaR for tail risk
```

### 2. Portfolio Risk Management
- **Position sizing**: Account for volatility clustering
- **Stop-loss**: Consider asymmetric downside risk
- **Hedging**: Time hedges during high-volatility regimes

### 3. Options Pricing
- **Volatility smile**: Fat tails justify out-of-money option premiums
- **Leverage effect**: Implies negative correlation between returns and volatility

## 🔬 Statistical Rigor

### Hypothesis Testing Framework
- **Significance level**: α = 5% throughout
- **Multiple testing**: Individual tests for each property
- **Robustness checks**: Multiple specifications for unit root tests

### Model Validation
- **Residual diagnostics**: White noise tests on ARMA residuals
- **ARCH effects**: Verify GARCH captures heteroskedasticity
- **Goodness-of-fit**: Information criteria (AIC, BIC)

## 📚 Theoretical Foundation

### Key References Implemented:
- **Mandelbrot (1963)**: Fat tails and scaling in financial returns
- **Engle (1982)**: ARCH modeling
- **Bollerslev (1986)**: GARCH specification
- **French & Roll (1986)**: Day-of-week effects
- **Zivot & Andrews (1992)**: Structural break unit root test
- **Lee & Strazicich (2003)**: Dual-break unit root test
- **Ng & Perron (2001)**: Modified AIC for lag selection

## 🚀 Future Extensions

1. **Multivariate Analysis**: DCC-GARCH for portfolio VaR
2. **Extreme Value Theory**: GPD for tail risk modeling
3. **Realized Volatility**: Incorporate intraday data
4. **Machine Learning**: LSTM for volatility forecasting
5. **Backtesting Framework**: Christoffersen tests for VaR accuracy
6. **Expected Shortfall**: CVaR as coherent risk measure

## 📊 Visualization Highlights

The analysis includes:
- **Time series plots**: Price, returns, rolling volatility
- **Correlograms**: ACF/PACF for returns and squared returns
- **Density plots**: Empirical vs. normal distribution
- **Leverage plots**: Price vs. volatility dual-axis
- **Seasonality plots**: Monthly/weekly pattern analysis
- **Unit root plots**: ZA test statistics across all break dates

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **Econometric testing**: Comprehensive hypothesis testing framework
- **Time series modeling**: ARMA, GARCH, structural breaks
- **Statistical software**: Advanced R programming
- **Financial theory**: Stylized facts of asset returns
- **Risk management**: Foundations for VaR/ES estimation
- **Research methodology**: From data to actionable insights

## 📄 Files

- `BAJI_VAR.Rmd`: Complete R Markdown analysis (1561 lines)
- `BAJI_VAR.pdf`: Compiled report with LaTeX formatting (49 pages)
- Data source: Yahoo Finance API via `yfR` package

## 👤 Author

**Mohamed Anas Baji**  
M2 IREF - Economic Risks and Data Science

## 📄 License

This project is academic coursework demonstrating advanced econometric analysis for financial risk management.

---

**Note**: This analysis provides the statistical foundation for Value at Risk modeling. All eight stylized facts are validated, confirming the need for sophisticated volatility and tail risk models in financial risk management. The presence of structural breaks and high volatility persistence has critical implications for dynamic hedging and capital allocation strategies.

## 🔑 Key Takeaway

**Standard normal-based VaR is inadequate** for this asset due to:
- Kurtosis > 6 (vs. 3 for normal)
- Volatility clustering (GARCH needed)
- Asymmetric tail risk (leverage effect)
- Structural instability (breaks in 2018, 2020, 2022)

**Recommended approach**: GARCH(1,1) with Student-t innovations + EVT for extreme quantiles.
