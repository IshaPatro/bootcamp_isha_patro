# NASDAQ Time Series Prediction

A time series analysis project that predicts NASDAQ market trends using historical data and technical indicators.

## Overview

This project analyzes NASDAQ historical data (2015-2025) to build predictive models for market forecasting. It incorporates technical indicators like moving averages, RSI, Bollinger Bands, and volatility metrics to enhance prediction accuracy.

**Target Users**: Data-driven investors, financial analysts, and researchers evaluating time series forecasting techniques.


## Data Storage

This stage demonstrates efficient data storage patterns using both CSV and Parquet formats with environment-driven configuration.

**Folder Structure:**
- `data/raw/` - CSV files with timestamped names
- `data/processed/` - Parquet files for optimized storage
- Environment variables control storage paths via `.env`

**Validation:** Files are reloaded and checked for shape consistency and proper data types. Parquet offers better compression and preserves data types automatically.

## Project Structure

```
├── src/stage05_data-storage_notebook.ipynb  # Storage implementation
├── data/
│   ├── raw/                    # CSV storage
│   └── processed/              # Parquet storage
└── README.md                   # Documentation
```
