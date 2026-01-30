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



### Performance Metrics
```
================================================================================
MODEL PERFORMANCE COMPARISON
================================================================================
Model           RMSE         MAE          MAPE        
--------------------------------------------------------------------------------
ARIMA+GARCH     $58.55       $44.08       6.53%
LSTM            $22.96       $17.43       2.62%
================================================================================


Based on the quantitative results, the comparative analysis reveals that LSTM substantially outperforms ARIMA+GARCH across all evaluated metrics. The LSTM model achieves an RMSE of $22.96 and MAPE of 2.62%, representing a **60.8% improvement in RMSE** and **59.9% improvement in MAPE** over the ARIMA+GARCH approach. The MAE improvement of **60.4%** further confirms LSTM's superior performance.
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

```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
```

### Installation
```bash
# Clone the repository
git clone https://github.com/M-Yassir/time-series-models-comparison.git
cd time-series-models-comparison
```

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

### Academic Supervision
This research was conducted under the supervision of **Mr. Abdessamad OUSAADANE** and **Mr. El Moukhtar ZEMMOURI**, whose expertise in quantitative finance and machine learning guided the methodology and analysis presented in this study.

### Data & Tools
- **Yahoo Finance**: For providing accessible and reliable financial data through their public API
- **Open-Source Community**: For developing and maintaining essential Python libraries including:
  - Data manipulation: pandas, numpy
  - Machine learning: TensorFlow/Keras, scikit-learn
  - Econometrics: statsmodels, pmdarima, arch
  - Visualization: matplotlib, seaborn

### Research Foundation
This work builds upon established literature in time series forecasting, econometric modeling, and financial machine learning.

## 📧 Contact

Abouchiba Mohamed Yassir - Email: m.abouchiba0@gmail.com

