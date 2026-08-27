import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import requests
from arch import arch_model

st.set_page_config(page_title="Volatility Tracker", layout="wide")
st.title("📈 Enterprise Volatility Analytics")

@st.cache_data(ttl=1800)
def fetch_stock_and_index_data(ticker: str, index_ticker: str = "^GSPC"):
    """Fetches 2-year history for both target stock and benchmark index."""
    data = yf.download([ticker, index_ticker], period="2y", progress=False)['Close']
    if data.empty or ticker not in data.columns:
        return None
    
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

def fetch_implied_volatility_polygon(ticker: str):
    """Fetches real-time At-The-Money (ATM) Implied Volatility via Polygon.io API."""
    try:
        api_key = st.secrets["POLYGON_API_KEY"]
    except Exception:
        return None, "Polygon API Key missing in Streamlit Secrets."

    try:
        # Fetch current stock price from Polygon
        snapshot_url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}?apiKey={api_key}"
        snap_res = requests.get(snapshot_url).json()
        
        if 'ticker' not in snap_res:
            return None, f"Ticker '{ticker}' not found on Polygon."
        
        current_price = snap_res['ticker']['day']['c']

        # Fetch front-month call options chain
        chain_url = (
            f"https://api.polygon.io/v3/snapshot/options/{ticker}"
            f"?contract_type=call&order=asc&sort=expiration_date&limit=100&apiKey={api_key}"
        )
        chain_res = requests.get(chain_url).json()

        results = chain_res.get('results', [])
        if not results:
            return None, "No active options contracts returned from Polygon."

        valid_contracts = []
        for contract in results:
            greeks = contract.get('greeks', {})
            iv = greeks.get('implied_volatility')
            strike = contract.get('details', {}).get('strike_price')
            exp = contract.get('details', {}).get('expiration_date')
            
            if iv and iv > 0 and strike:
                valid_contracts.append({
                    "iv": iv,
                    "strike": strike,
                    "expiration": exp,
                    "diff": abs(strike - current_price)
                })

        if not valid_contracts:
            return None, "No valid IV metrics found across front options contracts."

        # Find closest ATM strike
        valid_contracts.sort(key=lambda x: x['diff'])
        atm_contract = valid_contracts[0]

        return {
            "iv": float(atm_contract['iv']),
            "expiration": atm_contract['expiration'],
            "strike": float(atm_contract['strike']),
            "underlying_price": float(current_price)
        }, None

    except Exception as e:
        return None, f"Polygon API Error: {str(e)}"

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
    st.text_input("Benchmark Index:", value="^GSPC (S&P 500)", disabled=True)

if ticker_input:
    with st.spinner(f"Analyzing volatility and option chains for {ticker_input}..."):
        df = fetch_stock_and_index_data(ticker_input)
        iv_data, iv_error = fetch_implied_volatility_polygon(ticker_input)
        
        if df is None or len(df) < 30:
            st.error(f"Could not load sufficient market data for ticker '{ticker_input}'.")
        else:
            stock_30d_hv = df['Stock_30D_Vol'].iloc[-1]
            index_30d_hv = df['Index_30D_Vol'].iloc[-1]
            garch_30 = fit_garch(df['Stock_Return'])

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
