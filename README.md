# Predicting SPY Returns using RP-PCA and PCA Factors

This project investigates whether **RP-PCA (Risk-Premium PCA)** and **standard PCA factors** exhibit predictive power for future **SPY returns**, inspired by *Shi (2024)* and *Lettau & Pelger (2020)*.

We replicate and extend prior work by:
- Testing both **RP-PCA and PCA factors** (the original paper does not benchmark PCA)
- Evaluating **in-sample and out-of-sample predictive performance**
- Interpreting factors through **economic proxies (sentiment, growth, dispersion)**
- Exploring implications for **systematic trading strategies**

---

## Key Findings

- **Weak but consistent predictive signal** exists at medium horizons ($h=3,5$)
- **RP-PCA does NOT clearly outperform PCA** in predicting SPY returns
- **$RP-PCA_2$** and **$PC_3$** are highly correlated (~80%) and capture the **same latent signal**
- **PC$_5$** appears to capture a **distinct economic component** (linked to dispersion/energy)
- Out-of-sample performance shows:
  - Small but positive $R^2_{\text{OS}}$
  - Marginal Clark–West significance
  - Evidence of **regime dependence (post-2008 improvement)**

---

## Research Question

> Do RP-PCA factors exhibit similar predictive power for equity markets, and do they outperform traditional PCA factors?


---

## Methodology

### 1. Data
- **RP-PCA & PCA factors** from *Lettau & Pelger (2020)*
- Monthly data: **1963–2017**
- SPY returns (Yahoo Finance): **1993–2017**
- Final sample: **299 observations**

---

### 2. In-Sample Analysis

We estimate:

$r_{\text{SPY},t+h} = \beta_0 + \beta_1 F_{k,t} + \epsilon_t$

- Horizons: \(h = 1, 3, 5\)
- HAC (Newey–West) standard errors
- Evaluate:
  - t-stats
  - p-values
  - $R^2$

We also run:
- **Multivariate regressions**
- **Factor combinations**
- **Collinearity diagnostics**

---

### 3. Out-of-Sample Testing

- Expanding window (walk-forward)
- Benchmark: historical mean
- Metrics:
  - $R^2_{\text{OS}}$
  - MSPE
  - Clark–West test
  - Success ratio

---

### 4. Economic Interpretation

We construct factor proxies using sector ETFs:

- **Growth**: Cyclical − Defensive  
- **Sentiment**: High-beta − Low-beta  
- **Dispersion**: Cross-sector return variability  

We then analyze correlations and regressions between these proxies and PCA/RP-PCA factors.

---

### 5. Trading Strategy (Extension)

- Use predicted returns as signals
- Position sizing:
  - Binary
  - Proportional
  - Quantile-based
- Evaluate:
  - Sharpe
  - Drawdown
  - Turnover

---

## Results Summary

| Finding | Interpretation |
|--------|---------------|
| RP-PCA₂ significant | Captures macro/sentiment signal |
| PC₃ similar to RP-PCA₂ | Redundant signal |
| PC₅ survives multivariate tests | Distinct factor (likely dispersion/energy) |
| Weak $R^2$ | Signal exists but is small |
| Strong post-2008 performance | Regime dependence |

---

## Limitations

- Small dataset (~300 observations)
- Monthly frequency (limits signal granularity)
- Proxy-based economic factors
- No full real-time RP-PCA recomputation

---

## 🔮 Future Work

- Extend RP-PCA factors to recent data (post-2017)
- Test on **other assets (e.g., crude oil)**
- Incorporate **macro variables**:
  - Interest rates
  - Inflation
  - Credit spreads
  - VIX
- Regime-switching models
- Improved factor construction

---

## References

- Lettau, M., & Pelger, M. (2020). *Estimating latent asset-pricing factors*.  
- Shi, Q. (2024). *RP-PCA Thematic Investing Engine*. Prague Economic Papers.

