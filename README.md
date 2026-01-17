# Comparative Analysis: ARIMA+GARCH vs LSTM for Stock Price Forecasting

## 📊 Project Overview

This project presents a comprehensive comparative study between traditional econometric models (ARIMA+GARCH) and modern deep learning approaches (LSTM) for forecasting Meta Platforms Inc. (META) stock prices. The analysis spans from January 2023 to November 2025, capturing diverse market conditions and providing insights into the effectiveness of different modeling paradigms for financial time series.

## 🎯 Key Objectives

- **Compare Forecasting Performance**: Evaluate one-step-ahead forecasting accuracy of ARIMA+GARCH vs LSTM models
- **Volatility Modeling**: Implement GARCH for volatility clustering and uncertainty quantification
- **Modern vs Traditional**: Contrast interpretable econometric models with flexible deep learning approaches
- **Practical Insights**: Provide guidance on model selection for financial forecasting applications

## 📈 Methodology

### 1. **ARIMA+GARCH Model**
   - Auto-ARIMA for automatic parameter selection (ARIMA(0,0,0) identified)
   - GARCH(1,1) for volatility modeling and confidence intervals
   - Rolling window forecasting with retraining for each prediction
   - Statistical tests (ADF, KPSS) for stationarity validation

### 2. **LSTM Model**
   - 2-layer LSTM architecture with dropout regularization
   - 60-day window sequences for temporal pattern learning
   - MinMax scaling for neural network optimization
   - Fixed architecture evaluation (no retraining during test period)

## 🛠️ Technical Implementation

### Libraries Used
- **Data Handling**: pandas, numpy, yfinance
- **Visualization**: matplotlib, seaborn
- **Econometric Modeling**: statsmodels, pmdarima, arch
- **Deep Learning**: TensorFlow/Keras
- **Evaluation**: scikit-learn, scipy

### Key Features
- Automated ARIMA parameter selection with `pmdarima`
- Comprehensive stationarity testing (ADF, KPSS, ACF/PACF)
- Rolling window forecasting mimicking real-world conditions
- Confidence interval construction from GARCH volatility estimates
- Comparative visualization and statistical analysis

## 📊 Results Highlights

### Performance Metrics (Example Results)
```
ARIMA+GARCH: RMSE=$58.55, MAE=$44.08, MAPE=6.53%
LSTM:        RMSE=$22.75, MAE=$16.96, MAPE=2.56%
Improvement: 61.1% RMSE, 61.5% MAE, 60.8% MAPE
```

### Key Findings
1. **LSTM Superiority**: LSTM consistently outperforms ARIMA+GARCH across all metrics
2. **Random Walk Characteristics**: Auto-ARIMA selects ARIMA(0,0,0), suggesting efficient market behavior
3. **Volatility Insights**: GARCH provides valuable uncertainty quantification despite limited clustering
4. **Trade-offs**: ARIMA+GARCH offers interpretability; LSTM provides superior accuracy

## 📁 Project Structure

```
stock-forecasting-comparison/
├── Comparative_Analysis_ARIMA_GARCH_vs_LSTM.ipynb  # Main notebook
├── README.md                                       # This file
├── requirements.txt                                # Dependencies
└── data/                                          # Optional: Data directory
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/stock-forecasting-comparison.git
cd stock-forecasting-comparison


### Running the Analysis
```bash
# Launch Jupyter Notebook
jupyter notebook Comparative_Analysis_ARIMA_GARCH_vs_LSTM.ipynb
```

## 🔍 Key Insights for Practitioners

### When to Use ARIMA+GARCH:
- Need interpretable model parameters
- Require confidence intervals for risk management
- Limited computational resources
- Regulatory compliance requiring explainable models

### When to Use LSTM:
- Maximizing predictive accuracy is priority
- Working with large datasets
- Computational resources available
- Capturing complex non-linear patterns

## 📚 Academic Contributions

This study demonstrates:
1. The effectiveness of auto-ARIMA for financial time series
2. LSTM's ability to extract signals from seemingly random data
3. Practical implementation of rolling window forecasting
4. Comparative framework for evaluating financial forecasting models

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Yahoo Finance for providing stock data via `yfinance`
- Contributors to open-source libraries used in this analysis
- Academic literature on financial time series forecasting

## 📧 Contact

Abouchiba Mohamed Yassir - Email: m.abouchiba0@gmail.com

