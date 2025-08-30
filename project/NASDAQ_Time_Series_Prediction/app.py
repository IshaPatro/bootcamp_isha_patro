import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px

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
    fig = px.line(
        df, 
        x='Date', 
        y='Close/Last', 
        title='Interactive NASDAQ Historical Price Analysis'
    )

    fig.update_layout(
        template='plotly_white',
        plot_bgcolor="white", 
        paper_bgcolor="white",
        height = 800,
        title=dict(
            text="Interactive NASDAQ Historical Price Analysis",
            font=dict(size=22, color="black"),
            x=0.5,
            xanchor="center"
        ),
        xaxis_title="Date",
        yaxis_title="Closing Price (USD)",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date",
            showline=True,
            linecolor="black",
            tickfont=dict(color="black"),
            titlefont=dict(color="black"),
            gridcolor="lightgray"
        ),
        yaxis=dict(
            showline=True,
            linecolor="black",
            tickfont=dict(color="black"),
            titlefont=dict(color="black"),
            gridcolor="lightgray"
        )
    )
    
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
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    This chart displays the historical closing prices of the NASDAQ index.
    **Inference:** The visualization allows for dynamic exploration of long-term growth trends, volatility periods, and the impact of major market events by zooming into specific date ranges.
    """)

    st.subheader("NASDAQ Price Trend & Moving Averages")
    st.image(images['price_trend'], caption="NASDAQ Closing Price with 50-day and 200-day Moving Averages.", use_column_width=True)
    st.markdown("""
    ### Key Insight for Stakeholders
    The NASDAQ index shows clear trend patterns with periods of sustained growth interrupted by significant corrections. The crossover points between the 50-day and 200-day moving averages often signal important market regime changes that can guide investment timing decisions.
    ### Assumptions & Limitations
    - Past performance does not guarantee future results
    - Moving averages are lagging indicators and may not capture sudden market shifts
    - The analysis does not account for external factors like economic policy changes or global events
    """)

    st.subheader("Average Monthly NASDAQ Returns")
    st.image(images['monthly_returns'], caption="Seasonality in NASDAQ returns.", use_column_width=True)
    st.markdown("""
    ### Key Insight for Stakeholders
    The monthly returns analysis reveals clear seasonal patterns in NASDAQ performance. Certain months consistently show stronger returns, which can be leveraged for timing investment decisions and portfolio rebalancing.
    ### Assumptions & Limitations
    - Monthly averages may mask significant intra-month volatility
    - Historical seasonal patterns may not persist in the future due to changing market dynamics
    - The analysis does not account for specific events that may have influenced returns in particular years
    """)

    st.subheader("Comprehensive Time Series Analysis")
    st.image(images['time_series'], caption="Price, Daily Returns, Volatility, and RSI.", use_column_width=True)
    st.markdown("""
    **Inference:**
    - **Daily Returns:** The returns hover around zero, with periods of high volatility (volatility clustering). This is typical for financial assets.
    - **10-Day Rolling Volatility:** This chart quantifies the risk. Spikes in volatility correspond to periods of market uncertainty.
    - **RSI Indicator:** The Relative Strength Index (RSI) oscillates, indicating periods where the asset is potentially overbought (>70) or oversold (<30).
    """)

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
        
        **Inference:** The Linear Regression model captures the general trend but struggles with non-linear market movements, making it a basic baseline model.
        """)
    with col2:
        st.subheader("LSTM (Long Short-Term Memory) Model")
        st.image(images['lstm_prediction'], caption="Actual vs. Predicted Prices using LSTM.", use_column_width=True)
        st.markdown("""
        **Metrics:**
        - **RMSE:** 366.23
        - **MAE:** 283.05
        - **R2 Score:** 0.993

        **Inference:** The LSTM model tracks actual prices more closely, effectively capturing complex patterns due to its memory capabilities, making it superior for forecasting.
        """)
    
    st.subheader("Comparative Analysis of Models")
    st.markdown("""
    - **R2 Score (Coefficient of Determination):** The LSTM model has a slightly higher R2 Score (0.993) compared to the Linear Regression model (0.989). This indicates that the LSTM model explains approximately 99.3% of the variance in the NASDAQ price, making it marginally better in terms of explanatory power.
    - **Error Metrics (RMSE & MAE):** Contrary to the R2 score, the Linear Regression model shows significantly lower error metrics (RMSE: 236.71, MAE: 164.02) than the LSTM model (RMSE: 366.23, MAE: 283.05). A lower RMSE and MAE mean the average prediction error is smaller.
    
    **Conclusion:** While the LSTM model appears to follow the price curve more closely and has a better R2 score, the Linear Regression model is demonstrably more accurate in its predictions, with a substantially lower average error. This suggests that for this dataset, the simpler model provides a better balance of fit and accuracy, and the more complex LSTM model may be overfitting or requires further tuning.
    """)

    st.markdown("---")
    st.header("LSTM Model Performance Analysis")
    st.subheader("Scenario Comparison: Original vs. Simpler LSTM")
    st.image(images['lstm_comparison'], caption="Comparing two LSTM model configurations.", use_column_width=True)
    st.markdown("**Inference:** The simpler LSTM model shows a lower Root Mean Squared Error (RMSE), indicating that more complexity is not always better and that this model generalizes more effectively.")
    
    st.subheader("Distribution of Bootstrapped RMSE")
    st.image(images['bootstrap_dist'], caption="Assessing the stability of the LSTM model's performance.", use_column_width=True)
    st.markdown("**Inference:** Bootstrapping provides a 95% Confidence Interval for the model's error, adding statistical rigor and giving a reliable range for its expected performance.")

    st.markdown("---")
    st.header("Scenario & Sensitivity Analysis")
    st.subheader("Sensitivity of Risk-Adjusted Return")
    st.image(images['sensitivity'], caption="Tornado Chart for Sensitivity Analysis.", use_column_width=True)
    st.markdown("**Inference:** The tornado chart reveals that 'High Volatility' scenarios have the largest negative impact on risk-adjusted returns, highlighting volatility management as a key strategic focus.")

    st.markdown("---")
    st.header("Strategic Recommendations & Risks")
    st.subheader("Decision Implications")
    st.markdown("""
    Based on the analysis, the following actions are recommended:
    1.  **Strategic Timing**: Consider the seasonal patterns identified in monthly returns for portfolio rebalancing and entry/exit decisions.
    2.  **Risk Management**: Monitor the 10-day rolling volatility as an early warning indicator for potential spikes in risk.
    3.  **Diversification**: The sensitivity analysis shows significant impact from market regime changes, highlighting the importance of diversification across asset classes.
    4.  **Hedging Strategy**: During periods when the 50-day moving average crosses below the 200-day moving average, consider implementing hedging strategies to protect against potential downside risk.
    5.  **Regular Review**: Market conditions can change rapidly. Quarterly reviews of this analysis with updated data are recommended to ensure strategies remain aligned with current market dynamics.
    """)
    st.subheader("Assumptions & Risks")
    st.markdown("""
    - **Historical Patterns**: This analysis assumes that historical patterns provide meaningful insights for future market behavior.
    - **Market Efficiency**: It is assumed that markets are generally efficient but may experience periods of inefficiency that create opportunities.
    - **External Factors**: The analysis does not explicitly account for external shocks such as geopolitical events, policy changes, or black swan events.
    - **Data Quality**: The analysis is only as good as the underlying data. Any errors or biases in the data will affect the conclusions.
    - **Model Limitations**: Technical indicators like moving averages have known limitations and can generate false signals, particularly in choppy markets.
    """)

