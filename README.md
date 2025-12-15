# Stock Price Forecasting Project

## 📈 Project Overview
Compare ARIMA+GARCH and LSTM models for predicting stock prices.

## 🛠️ Models Used
- **ARIMA(0,0,0) + GARCH(1,1)**: Traditional statistical model
- **LSTM**: Deep learning neural network

## 📊 Key Features
- Forecasts stock prices using both statistical and AI methods
- Provides 95% confidence intervals for ARIMA+GARCH forecasts
- Visual comparison of both models
- Easy to modify and extend

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install numpy pandas matplotlib scikit-learn statsmodels arch tensorflow
```

### 2. Prepare Your Data
- Save your stock price data as `data.csv`
- Should have a date column and a price column

### 3. Run the Models
Open `forecast.ipynb` in Jupyter and run all cells.

## ⚙️ Model Configuration

### ARIMA+GARCH Settings
```python
TRAIN_SIZE = 0.8  # 80% for training, 20% for testing
ARIMA_ORDER = (0, 0, 0)  # Based on ACF/PACF analysis
GARCH_ORDER = (1, 1)     # Standard for volatility
```

### LSTM Settings
```python
WINDOW_SIZE = 60    # Lookback period
LSTM_UNITS = 50     # Neural network complexity
EPOCHS = 100        # Training iterations
```

## 📈 What You'll See
1. **Training Fit**: How well models learned historical patterns
2. **Test Forecasts**: Predictions on unseen data
3. **Confidence Intervals**: Uncertainty range for ARIMA+GARCH
4. **Model Comparison**: Side-by-side performance evaluation

## 🔧 Tips for Better Results
1. **Check ACF/PACF**: Validate ARIMA order selection
2. **Try different LSTM settings**: Window size, units, layers
3. **Add more features**: Volume, indicators for LSTM
4. **Experiment with models**: Try ARIMA(1,0,0) or GARCH variants

## 📁 Files
- `forecast.ipynb` - Main notebook with all code
- `data.csv` - Your price data
- `README.md` - This file

## 🎯 Results Expected
- Reasonable price forecasts for both models
- ARIMA+GARCH with confidence bands
- LSTM capturing complex patterns
- Visual comparison of performance

## 💡 Pro Tips
- ARIMA+GARCH is better for volatility estimation
- LSTM can capture complex non-linear patterns
- Consider combining both for ensemble forecasts
- Always validate on out-of-sample data

---

**Ready to start?** Open the notebook and run the first cell!
