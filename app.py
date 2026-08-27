import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from arch import arch_model

st.set_page_config(page_title="Volatility Tracker", layout="wide")
st.title("📈 Enterprise Volatility Analytics")

@st.cache_data(ttl=1800)  # Cache market data for 30 minutes to improve speed
def fetch_stock_and_index_data(ticker: str, index_ticker: str = "^GSPC"):
    """Fetches 2-year history for both target stock and benchmark index."""
    # Download batch data for stock and S&P 500
    data = yf.download([ticker, index_ticker], period="2y", progress=False)['Close']
    if data.empty or ticker not in data.columns:
        return None, None
    
    df = pd.DataFrame()
    df['Stock_Close'] = data[ticker]
    df['Index_Close'] = data[index_ticker]
    
    # Calculate Daily Log Returns
    df['Stock_Return'] = np.log(df['Stock_Close'] / df['Stock_Close'].shift(1))
    df['Index_Return'] = np.log(df['Index_Close'] / df['Index_Close'].shift(1))
    df = df.dropna()
    
    # 30-Day Realized Volatilities (Annualized)
    df['Stock_30D_Vol'] = df['Stock_Return'].rolling(window=30).std() * np.sqrt(252)
    df['Index_30D_Vol'] = df['Index_Return'].rolling(window=30).std() * np.sqrt(252)
    
    # 30-Day Rolling Beta relative to S&P 500
    rolling_cov = df['Stock_Return'].rolling(window=30).cov(df['Index_Return'])
    rolling_var = df['Index_Return'].rolling(window=30).var()
    df['Rolling_Beta'] = rolling_cov / rolling_var
    
    return df

def fetch_implied_volatility(ticker: str):
    """Scans nearest option expiration cycles to pull At-The-Money (ATM) Implied Volatility."""
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return None, "No option chain available for this symbol."
        
        # Grab target stock current price
        history = tk.history(period="5d")
        if history.empty:
            return None, "Could not fetch current price."
        current_price = history['Close'].iloc[-1]

        # Scan the front-month contract expiration (~30 days target)
        target_exp = expirations[0]
        opt_chain = tk.option_chain(target_exp)
        calls = opt_chain.calls
        
        if calls.empty:
            return None, "Empty options chain."

        # Find closest At-The-Money (ATM) strike
        calls['strike_diff'] = (calls['strike'] - current_price).abs()
        atm_call = calls.sort_values('strike_diff').iloc[0]
        
        iv = atm_call['impliedVolatility']
        strike = atm_call['strike']
        
        if pd.isna(iv) or iv == 0:
            return None, "Implied Volatility data unavailable for ATM strike."
            
        return {
            "iv": float(iv),
            "expiration": target_exp,
            "strike": float(strike),
            "underlying_price": float(current_price)
        }, None
    except Exception as e:
        return None, f"Error processing options: {str(e)}"

def fit_garch(returns: pd.Series, horizon: int = 30) -> float:
    """Projects forward 30-day volatility using GARCH(1,1)."""
    scaled_returns = returns * 100
    model = arch_model(scaled_returns, vol='GARCH', p=1, q=1, mean='constant', dist='normal')
    model_fit = model.fit(disp='off')
    forecast = model_fit.forecast(horizon=horizon)
    avg_daily_var = forecast.variance.iloc[-1].mean()
    return float((np.sqrt(avg_daily_var) / 100) * np.sqrt(252))

# --- Dashboard Layout ---
col_input, col_bench = st.columns([2, 2])
with col_input:
    ticker_input = st.text_input("Enter Equity Ticker:", value="AAPL").upper().strip()
with col_bench:
    index_input = st.text_input("Benchmark Index:", value="^GSPC (" + "S&P 500" + ")", disabled=True)

if ticker_input:
    with st.spinner(f"Analyzing volatility and option chains for {ticker_input}..."):
        df = fetch_stock_and_index_data(ticker_input)
        iv_data, iv_error = fetch_implied_volatility(ticker_input)
        
        if df is None or len(df) < 30:
            st.error(f"Could not load sufficient market data for ticker '{ticker_input}'.")
        else:
            stock_30d_hv = df['Stock_30D_Vol'].iloc[-1]
            index_30d_hv = df['Index_30D_Vol'].iloc[-1]
            garch_30 = fit_garch(df['Stock_Return'])
            current_beta = df['Rolling_Beta'].iloc[-1]

            # Top Row KPI Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("30D Realized Vol (Stock)", f"{stock_30d_hv:.2%}")
            m2.metric("30D Realized Vol (S&P 500)", f"{index_30d_hv:.2%}")
            m3.metric("30D Forward GARCH Forecast", f"{garch_30:.2%}")
            
            if iv_data:
                m4.metric(
                    "ATM Implied Volatility (IV)", 
                    f"{iv_data['iv']:.2%}",
                    help=f"ATM Strike ${iv_data['strike']} (Exp: {iv_data['expiration']})"
                )
            else:
                m4.metric("Option Implied Volatility (IV)", "N/A", delta=iv_error, delta_color="off")

            st.markdown("---")

            # Chart 1: Stock Volatility vs. S&P 500 Volatility
            st.subheader(f"Historical 30-Day Volatility: {ticker_input} vs. S&P 500")
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=df.index, y=df['Stock_30D_Vol'], mode='lines', name=f'{ticker_input} 30D Vol', line=dict(width=2)))
            fig_vol.add_trace(go.Scatter(x=df.index, y=df['Index_30D_Vol'], mode='lines', name='S&P 500 30D Vol', line=dict(dash='dash', color='gray')))
            fig_vol.update_layout(
                xaxis_title="Date",
                yaxis_title="Annualized Volatility",
                yaxis_tickformat='.0%',
                template="plotly_white",
                height=450
            )
            st.plotly_chart(fig_vol, use_container_width=True)

            # Chart 2: Rolling Beta relative to S&P 500
            st.subheader(f"30-Day Rolling Beta ({ticker_input} relative to S&P 500)")
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(x=df.index, y=df['Rolling_Beta'], mode='lines', name='Beta', line=dict(color='purple')))
            fig_beta.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Market Beta (1.0)")
            fig_beta.update_layout(
                xaxis_title="Date",
                yaxis_title="Beta Coefficient",
                template="plotly_white",
                height=350
            )
            st.plotly_chart(fig_beta, use_container_width=True)
