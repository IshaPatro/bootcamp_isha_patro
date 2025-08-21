# NASDAQ Time Series Prediction

A time series analysis project that predicts NASDAQ market trends using historical data and technical indicators.

## Overview

This project analyzes NASDAQ historical data (2015-2025) to build predictive models for market forecasting. It incorporates technical indicators like moving averages, RSI, Bollinger Bands, and volatility metrics to enhance prediction accuracy.

**Target Users**: Data-driven investors, financial analysts, and researchers evaluating time series forecasting techniques.

## Features

- **Data Processing**: Smart date handling and strategic null value management
- **Technical Indicators**: Moving averages, RSI, Bollinger Bands, volatility analysis
- **Time Series Analysis**: Daily and log returns, trend identification
- **Clean Pipeline**: Automated data processing with comprehensive validation

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run analysis**
   Open `nasdaq_time_series.ipynb` and execute all cells

## Dataset

- **Period**: August 2015 - August 2025
- **Records**: 2,517 trading days
- **Features**: 16 columns including OHLC prices and technical indicators

## Technical Approach

- **Null Handling**: Domain-specific strategies (returns=0, volatility=0.01, RSI=50)
- **Date Processing**: Proper datetime conversion and chronological sorting
- **Feature Engineering**: Comprehensive technical indicator calculation
- **Data Quality**: 100% complete dataset after strategic preprocessing

## Project Structure

```
├── nasdaq_time_series.ipynb    # Main analysis notebook
├── data/
│   ├── HistoricalData.csv      # Raw NASDAQ data
│   └── processed/              # Cleaned dataset
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## Next Steps

- Implement machine learning models (LSTM, ARIMA)
- Add backtesting and performance evaluation
- Develop interactive visualizations

---

**Disclaimer**: This project is for educational purposes. Not intended for live trading decisions.

**Contact**: [GitHub](https://github.com/IshaPatro/NASDAQ_Time_Series_Prediction)