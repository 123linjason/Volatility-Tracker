import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from arch import arch_model

st.set_page_config(page_title="Volatility Tracker", layout="wide")
st.title("📈 Internal Volatility Analytics")

def fetch_and_prep_data(ticker: str):
    tk = yf.Ticker(ticker)
    df = tk.history(period="2y")
    if df.empty:
        return None, None, None
    
    # Daily log returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()
    
    # Realized Volatilities
    df['30D_Realized'] = df['Log_Return'].rolling(window=30).std() * np.sqrt(252)
    df['60D_Realized'] = df['Log_Return'].rolling(window=60).std() * np.sqrt(252)
    
    # Implied Volatility (ATM Call, Front Expiration)
    implied_vol = np.nan
    if tk.options:
        try:
            opt = tk.option_chain(tk.options[0])
            calls = opt.calls
            current_price = df['Close'].iloc[-1]
            atm_row = calls.iloc[(calls['strike'] - current_price).abs().argmin()]
            implied_vol = float(atm_row['impliedVolatility'])
        except Exception:
            pass

    return df, implied_vol, tk

def fit_garch(returns: pd.Series, horizon: int = 30) -> float:
    scaled_returns = returns * 100
    model = arch_model(scaled_returns, vol='GARCH', p=1, q=1, mean='constant', dist='normal')
    model_fit = model.fit(disp='off')
    forecast = model_fit.forecast(horizon=horizon)
    avg_daily_var = forecast.variance.iloc[-1].mean()
    return float((np.sqrt(avg_daily_var) / 100) * np.sqrt(252))

ticker_input = st.text_input("Enter Equity Ticker:", value="AAPL").upper().strip()

if ticker_input:
    with st.spinner(f"Fetching market data for {ticker_input}..."):
        df, iv, tk = fetch_and_prep_data(ticker_input)
        
        if df is None or len(df) < 30:
            st.error(f"Could not load sufficient market data for ticker '{ticker_input}'.")
        else:
            current_hv_30 = df['30D_Realized'].iloc[-1]
            garch_30 = fit_garch(df['Log_Return'])

            m1, m2, m3 = st.columns(3)
            m1.metric("30-Day Realized Volatility", f"{current_hv_30:.2%}")
            m2.metric("Forward GARCH(1,1) Forecast (30D)", f"{garch_30:.2%}")
            m3.metric("Option Implied Volatility (IV)", f"{iv:.2%}" if not np.isnan(iv) else "N/A")

            st.subheader("Historical Realized Volatility Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['30D_Realized'], mode='lines', name='30D Realized Vol'))
            fig.add_trace(go.Scatter(x=df.index, y=df['60D_Realized'], mode='lines', name='60D Realized Vol', line=dict(dash='dash')))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Annualized Volatility",
                yaxis_tickformat='.0%',
                template="plotly_white",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
