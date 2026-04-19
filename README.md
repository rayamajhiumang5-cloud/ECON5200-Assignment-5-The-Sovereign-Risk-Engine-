# ECON 5200 — Assignment 5: The Sovereign Risk Engine

A four-phase machine learning pipeline that builds a sovereign economic crisis early-warning system using World Bank WDI data. The project moves from raw data ingestion through regularization, classification, evaluation, and cost-sensitive deployment optimization.

---


## Overview
This notebook simulates an IMF analyst workflow: given macroeconomic indicators for ~150 countries, can we predict which will experience a GDP per capita contraction? The pipeline answers this in four progressive phases, each building on the last.

---
## Phase 1 : The Complexity Trap: OLS Failure and Regularization Rescue

Demonstrates why ordinary least squares breaks down when the number of predictors approaches the number of observations (high p/n ratio). Applies Ridge and Lasso regression via cross-validated hyperparameter search and compares out-of-sample performance. Traces the Lasso coefficient path to identify which indicators enter the model first and discusses what it means for a predictor to be zeroed out.

---

## Phase 2 :The Crisis Classifier: From Forecasting to Classification

Defines the binary outcome (crisis = negative GDP per capita growth) and exposes the failure of the Linear Probability Model when predicted probabilities fall outside [0, 1]. Replaces it with logistic regression, interprets coefficients as odds ratios, and visualizes the sigmoid vs. linear fit side by side.

---

## Phase 3: Operational Deployment: Metrics That Matter

Addresses the accuracy paradox in imbalanced classification. Builds a full evaluation suite — confusion matrix, classification report, ROC curve, and Precision-Recall curve — and explains why PR-AUC is more informative than ROC-AUC for rare-event detection. Performs threshold analysis under a capacity constraint (max 5 missions per quarter) and contrasts it with the F1-optimal threshold.

---

## Phase 4: AI Context Engineering (The P.R.I.M.E. Framework)

Uses structured AI prompting to generate and run two advanced analyses:

* **Bootstrap Lasso Stability**: 200 resamples to measure how consistently each indicator is selected, separating robust signals from fragile correlated proxies.

* **Cost-Sensitive Threshold Optimization**:  Sweeps τ using an asymmetric cost structure ($50B for a missed crisis vs. $2M for a false alarm) and produces a three-way comparison against the F1-optimal and capacity-constrained thresholds.

---

## Data

* Source: World Bank Development Indicators (WDI) via wbgapi


* Coverage: ~150 countries, 2013–2019 (country-level means)

* Indicators: 30+ variables spanning macroeconomics, trade, demographics, health, education, governance, and finance

* Target: Binary crisis indicator (1 if GDP per capita growth < 0)

---

## Requirements
```
pip install wbgapi scikit-learn statsmodels matplotlib seaborn numpy pandas
```
---

## Key Outputs


| Phase         | Output|
| ------------- |:-------------:|
| 1.            | OLS vs. Ridge vs. Lasso comparison table; Lasso coefficient path plot   |
| 2.            | LPM vs. Logistic side-by-side visualization; odds ratio table     |
| 3.            | Confusion matrix; ROC and PR curves; threshold analysis plot     |
| 4.            | Bootstrap selection frequency bar chart; cost curve with annotated optimal threshold; three-way threshold comparison table |

---
