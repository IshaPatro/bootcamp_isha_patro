import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(layout="wide", page_title="NASDAQ Analysis Dashboard", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main {
    background-color: #FFFFFF;
    color: #0E1117;
}
.css-1d391kg {
    display: none;
}
h1, h2, h3, h4, h5, h6 {
    color: #0E1117;
}
.st-emotion-cache-16txtl3 {
    color: #0E1117;
}
.st-expander {
    border-color: #CCCCCC !important;
    border-radius: 0.5rem;
}
.st-expander header {
    color: #0E1117 !important;
    background-color: #F0F2F6;
}
hr {
    border-top: 1px solid #0E1117 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        images = {
            'price_trend': Image.open('deliverables/images/nasdaq_price_trend.png'),
            'time_series': Image.open('deliverables/images/nasdaq_time_series_analysis.png'),
            'monthly_returns': Image.open('deliverables/images/monthly_returns.png'),
            'linear_regression': Image.open('deliverables/images/actual_vs_predicted_linear_regression.png'),
            'lstm_prediction': Image.open('deliverables/images/nasdaq_lstm_prediction.png'),
            'lstm_comparison': Image.open('deliverables/images/nasdaq_original_vs_simpler_lstm.png'),
            'bootstrap_dist': Image.open('deliverables/images/nasdaq_distribution_of_bootstrap.png'),
            'sensitivity': Image.open('deliverables/images/tornado_sensitivity.png')
        }
        
        df = pd.read_csv('data/HistoricalData.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        if df['Close/Last'].dtype == 'object':
             df['Close/Last'] = df['Close/Last'].replace({'\$': '', ',': ''}, regex=True).astype(float)
        df = df.sort_values('Date')

        return df, images
    except FileNotFoundError as e:
        st.error(f"Error loading data or images: {e}. Make sure 'data/HistoricalData.csv' and the image files in 'deliverables/images/' exist.")
        return None, None

df, images = load_data()

if df is not None and images is not None:
    st.title("NASDAQ Financial Analysis Report")
    st.header("Executive Summary")
    st.markdown("""
    This dashboard provides a comprehensive analysis of the NASDAQ index, covering historical performance, technical indicators, and predictive modeling. The analysis indicates a strong long-term uptrend, though subject to periods of significant volatility.
    **Key Findings:**
    - **Trend:** The NASDAQ exhibits a clear upward trend, consistently trading above its 50-day and 200-day moving averages, which is a bullish signal.
    - **Seasonality:** The index shows distinct monthly performance patterns, with November and July being historically strong, while September shows notable weakness.
    - **Volatility:** Daily returns are characterized by volatility clustering. The 10-day rolling volatility indicates periods of high and low risk.
    - **Modeling:** While both models show high R2 scores, the Linear Regression model has a significantly lower error rate (RMSE, MAE), suggesting it provides more accurate predictions than the LSTM model in this scenario.
    - **Risk:** Sensitivity analysis reveals that the portfolio's risk-adjusted returns are most negatively impacted by high-volatility scenarios.

    **Recommendation:**
    The analysis supports a bullish long-term outlook. However, investors should remain cautious of short-term volatility. Contrary to initial visual impressions, the simpler Linear Regression model is the more reliable predictive tool based on key error metrics.
    """)

    st.markdown("---")
    st.header("Market Overview & Technical Analysis")
    st.subheader("NASDAQ Price Trend")

    # --- Matplotlib Plot ---
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['Date'], df['Close/Last'], color='blue', label='NASDAQ Close Price')
    ax.set_title("NASDAQ Historical Closing Prices", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price (USD)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    This chart displays the historical closing prices of the NASDAQ index.
    **Inference:** The visualization allows for dynamic exploration of long-term growth trends, volatility periods, and the impact of major market events.
    """)

    st.subheader("NASDAQ Price Trend & Moving Averages")
    st.image(images['price_trend'], caption="NASDAQ Closing Price with 50-day and 200-day Moving Averages.", use_column_width=True)
    st.subheader("Average Monthly NASDAQ Returns")
    st.image(images['monthly_returns'], caption="Seasonality in NASDAQ returns.", use_column_width=True)
    st.subheader("Comprehensive Time Series Analysis")
    st.image(images['time_series'], caption="Price, Daily Returns, Volatility, and RSI.", use_column_width=True)

    st.markdown("---")
    st.header("Predictive Modeling Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Linear Regression Model")
        st.image(images['linear_regression'], caption="Actual vs. Predicted Prices using Linear Regression.", use_column_width=True)
        st.markdown("""
        **Metrics:**
        - **RMSE:** 236.71
        - **MAE:** 164.02
        - **R2 Score:** 0.989
        """)
    with col2:
        st.subheader("LSTM Model")
        st.image(images['lstm_prediction'], caption="Actual vs. Predicted Prices using LSTM.", use_column_width=True)
        st.markdown("""
        **Metrics:**
        - **RMSE:** 366.23
        - **MAE:** 283.05
        - **R2 Score:** 0.993
        """)

    st.subheader("Comparative Analysis of Models")
    st.markdown("""
    While the LSTM model has a slightly higher R2 score, the Linear Regression model shows lower RMSE and MAE, suggesting it provides more accurate predictions for this dataset.
    """)

    st.subheader("Scenario & Sensitivity Analysis")
    st.image(images['sensitivity'], caption="Tornado Chart for Sensitivity Analysis.", use_column_width=True)
