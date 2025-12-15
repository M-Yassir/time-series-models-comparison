# Time Series Forecasting with ARIMA+GARCH and LSTM

## 📊 Project Overview
This project implements and compares two advanced time series forecasting models for financial data:
1. **ARIMA(0,0,0) + GARCH(1,1)**: A traditional econometric approach combining mean and volatility modeling
2. **LSTM**: A deep learning approach using Long Short-Term Memory neural networks

The models are trained on historical price data and evaluated on test data with visualization of forecasts and confidence intervals.

## 🏗️ Model Architectures

### ARIMA+GARCH Model
- **ARIMA(0,0,0)**: Models the mean of log returns (no autoregressive or moving average components)
- **GARCH(1,1)**: Models volatility clustering in residuals
- **Key Features**:
  - Models log returns instead of raw prices
  - Provides 95% confidence intervals based on volatility forecasts
  - Online updating during test period
  - Proper scaling/descaling of returns for GARCH compatibility

### LSTM Model
- **Architecture**: Single-layer LSTM with dropout for regularization
- **Features**: 
  - Sequence-to-one prediction
  - Min-max scaling for normalization
  - Early stopping to prevent overfitting
  - Optimized for univariate time series

## 📁 Project Structure
```
project/
├── arima_garch_lstm.ipynb     # Main Jupyter notebook
├── data/                      # Data directory
│   └── your_data.csv          # Historical price data
├── models/                    # Saved models
├── README.md                  # This file
└── requirements.txt           # Dependencies
```

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd <project-directory>

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
statsmodels>=0.12.0
arch>=5.0.0
tensorflow>=2.6.0
jupyter>=1.0.0
```

## 🚀 Usage

### 1. Prepare Your Data
- Place your time series data in CSV format in the `data/` directory
- Ensure the CSV has a date column and a price column
- Data should be in chronological order

### 2. Run the Analysis
```bash
# Launch Jupyter notebook
jupyter notebook arima_garch_lstm.ipynb
```

### 3. Configure Parameters
In the notebook, modify these key parameters:

```python
# Data parameters
TRAIN_TEST_SPLIT = 0.8  # 80% training, 20% testing
WINDOW_SIZE = 60        # Lookback period for LSTM

# ARIMA+GARCH parameters
ARIMA_ORDER = (0, 0, 0)  # Based on ACF/PACF analysis
GARCH_ORDER = (1, 1)     # Standard for financial data

# LSTM parameters
LSTM_UNITS = 50
DROPOUT_RATE = 0.2
EPOCHS = 100
BATCH_SIZE = 32
```

### 4. Key Code Sections

#### ARIMA+GARCH Implementation
```python
# Log returns calculation
train_log_returns = np.log(training_data / training_data.shift(1)).dropna()

# ARIMA model
arima_model = ARIMA(train_log_returns, order=(0,0,0))
arima_fit = arima_model.fit()

# GARCH on residuals
garch_model = arch_model(arima_residuals, vol='Garch', p=1, q=1, dist='normal')
garch_fit = garch_model.fit(disp="off")
```

#### LSTM Implementation
```python
# Sequence creation
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data)-window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# LSTM model architecture
model = Sequential([
    LSTM(units=50, return_sequences=False, input_shape=(window_size, 1)),
    Dropout(0.2),
    Dense(1)
])
```

## 📈 Outputs & Visualizations

The notebook generates three main visualizations:

1. **Combined Forecast Plot**: Side-by-side comparison of both models
2. **ARIMA+GARCH Detailed Plot**: Shows training fit, test forecast, and 95% confidence intervals
3. **Test Period Comparison**: Zoomed view comparing both models against actual prices

### Sample Metrics
```
ARIMA+GARCH Metrics:
- Training Mean Return: 0.02% per period
- Volatility Persistence: 0.95
- Average Forecast Volatility: 1.5%

LSTM Metrics:
- Training MAE: 0.85
- Test MAE: 1.23
- Training Time: 45 seconds
```

## 🔧 Model Tuning

### ARIMA+GARCH Tuning
- Check ACF/PACF plots to validate ARIMA order
- Consider different GARCH variants (EGARCH, GJR-GARCH) for asymmetric volatility
- Adjust scaling factor if convergence issues occur

### LSTM Tuning
- Experiment with different window sizes (10-100)
- Try bidirectional LSTM for capturing past and future context
- Adjust number of layers and units
- Modify dropout rate to prevent overfitting

## ⚠️ Common Issues & Solutions

### Issue 1: ARIMA+GARCH Forecast Explodes
**Solution**: The code includes a damping factor to prevent exponential growth. If issues persist:
- Check if training mean return is reasonable
- Verify data stationarity
- Consider adding a mean reversion term

### Issue 2: LSTM Overfitting
**Solution**:
- Increase dropout rate
- Add L2 regularization
- Reduce model complexity
- Use more training data

### Issue 3: GARCH Convergence Issues
**Solution**:
- Scale returns by 100 (already implemented)
- Try different optimization algorithms
- Check for outliers in returns

## 📊 Model Comparison

| Aspect | ARIMA+GARCH | LSTM |
|--------|-------------|------|
| **Interpretability** | High - clear parameters | Low - black box |
| **Volatility Modeling** | Explicit via GARCH | Implicit in weights |
| **Confidence Intervals** | Yes, statistically derived | No, requires Monte Carlo |
| **Training Speed** | Fast (seconds) | Slow (minutes) |
| **Data Requirements** | Lower | Higher |
| **Non-linearity Capture** | Limited | Excellent |

## 🎯 Best Practices

1. **Always check residuals**: Ensure no autocorrelation remains in ARIMA residuals
2. **Validate stationarity**: Use ADF test before modeling
3. **Cross-validate**: Use time series cross-validation for robust evaluation
4. **Monitor overfitting**: Compare training vs test performance
5. **Consider ensemble**: Combine both models for potentially better performance

## 📚 References

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control*
2. Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation
3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory
4. Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity

## 👥 Authors

- [Your Name/Team Name]
- Contact: [Your Email]

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Statsmodels and Arch package developers
- TensorFlow/Keras team
- Financial data providers

---

**Note**: This is a template README. Replace bracketed `[ ]` information with your specific project details. For production use, ensure proper validation, error handling, and security measures are implemented.
