# Magnificent 7 Stock Price Prediction

**Course:** DSC 223
**Team:** Sin U Choi & Joshua Guo

---

## Overview

This project applies Ridge Regression to predict next-day stock returns for the **Magnificent 7 tech stocks** (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA) using technical indicators derived from historical OHLCV data fetched via `yfinance`.

The core research question is: **does training a single model on all 7 stocks together outperform training separate models for each stock?** We compare both strategies and evaluate which generalizes better out-of-sample.

---

## Repository Structure

```text
├── config.py                               # All settings — edit this file to change tickers/dates
├── Final.py                                # Single-ticker analysis with 5-day forecast
├── Intergrate_model.py                     # Combined Mag 7 Ridge model (Strategy B)
├── Compare.py                              # Head-to-head comparison of both strategies
├── requirements.txt                        # Python dependencies
└── methodology_presentation_final_v1.pptx  # Project presentation slides
```

---

## Dataset

- **Source:** Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance)
- **Tickers:** AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- **Date range:** 2023-01-01 → 2026-01-01 (configurable in `config.py`)
- **Input features (22 total):**
  - EMA distance from price (9, 21, 50, 200-day)
  - RSI (14-day), RSI distance from 50, RSI 3-day change
  - MACD, MACD signal cross, MACD histogram change (normalized by price)
  - Rolling returns: 1, 3, 5, 10, 20-day
  - Rolling volatility: 5, 10, 20-day standard deviation
  - Volatility regime (vol_5 / vol_20 ratio)
  - 20-day z-score of price
  - Volume ratio vs. 20-day average
  - High-low range normalized by close
- **Target:** Next-day return (percentage change), predicted via regression
- **Preprocessing:** IQR clipping (1.5×) on all return/volatility features to handle outliers; 80/20 chronological train/test split with no shuffling to prevent data leakage

---

## Models

### Strategy A — Single-Ticker Ridge Regression (`Final.py`)

A Ridge Regression model trained independently on one stock at a time. Also produces a 5-day forward price forecast using iterative prediction.

### Strategy B — Combined Mag 7 Ridge Regression (`Intergrate_model.py`)

One Ridge model trained on all 7 stocks pooled together, then evaluated per-ticker on the out-of-sample 20%. Uses price-relative features so patterns are comparable across tickers regardless of absolute price level.

### Strategy Comparison (`Compare.py`)

Head-to-head evaluation of Strategy A vs B across all 7 tickers. Reports MAE and R² per ticker and declares an overall winner based on which strategy wins more matchups.

**Why Ridge over plain Linear Regression?** The feature set includes correlated indicators (e.g. multiple EMAs). Ridge's L2 regularization reduces variance from multicollinearity without dropping features.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure settings

Edit `config.py` to change tickers, date range, or indicator parameters:

```python
TICKER        = "AAPL"        # For single-ticker analysis
START_DATE    = "2023-01-01"
END_DATE      = "2026-01-01"
FORECAST_DAYS = 5
```

### 3. Run the scripts

```bash
# Single-ticker analysis + 5-day forecast
python3 Final.py

# Combined Mag 7 model
python3 Intergrate_model.py

# Strategy A vs B comparison
python3 Compare.py
```

---

## Example Output

**`Intergrate_model.py`** prints per-ticker in-sample and out-of-sample MAE and R², and saves:

- `Integrate_model_predictions.png` — Actual vs. predicted price for all 7 tickers
- `Integrate_model_features.png` — Top 15 feature importances by Ridge coefficient magnitude

**`Compare.py`** saves:

- `comparison.png` — Side-by-side Strategy A vs B prediction plots per ticker
- `comparison_summary.png` — Bar charts of MAE and R² across all 7 tickers

---

## Decisions & Trade-offs

| Decision | Reasoning |
| --- | --- |
| Ridge over Linear Regression | Multiple EMAs are correlated; L2 regularization reduces overfitting from multicollinearity |
| Chronological 80/20 split, no shuffle | Prevents future data leaking into training — standard for time series evaluation |
| IQR clipping on return/volatility features | Stock data has fat tails; clipping prevents extreme outliers from dominating the regression |
| Price-relative features (e.g. EMA distance %) | Makes features cross-ticker comparable so the combined model can generalize across stocks |
| Predicting returns instead of price | Returns are more stationary and scale-invariant than raw prices |
| Did not implement deep learning (LSTM/Transformer) | Would likely capture temporal dependencies better but was out of scope given project timeline |

---

## Team Contributions

All major design decisions — including feature engineering, the single vs. combined model comparison framework, and evaluation methodology — were made jointly through discussion.

- **Sin U Choi:** [e.g. feature engineering, Compare.py, visualizations, README]
- **Joshua Guo:** [e.g. Final.py, Intergrate_model.py, config architecture, presentation slides]

> Because most decisions were discussed and iterated on together, the above reflects primary ownership rather than exclusive authorship.

---

## AI Usage

AI tools (Claude, ChatGPT) were used during this project for debugging assistance, boilerplate code generation, and README writing. All model design decisions, feature selection, data pipeline architecture, and analysis interpretation were made by the team.

---

## Citations

- **Data:** Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance)
- **Ridge Regression:** scikit-learn — [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- **Technical indicators (EMA, RSI, MACD):** [Investopedia](https://www.investopedia.com)

---

## Requirements

```text
yfinance>=0.2.40
pandas>=2.2.0
matplotlib>=3.9.0
scikit-learn>=1.5.0
numpy>=1.26.0
```
