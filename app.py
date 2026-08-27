import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from arch import arch_model

st.set_page_config(page_title="Enterprise Volatility Analytics", layout="wide")

# Top Navigation Header
st.title("📈 Enterprise Volatility Analytics")

# --- UI User Guide & Operational Reference ---
with st.expander("📖 Dashboard User Guide & Operational Reference", expanded=False):
    st.markdown("### Executive Overview")
    st.write(
        "This dashboard provides a real-time framework for comparing underlying equity historical volatility "
        "against options market implied volatility (IV), forward GARCH projections, and systematic risk metrics."
    )
    
    st.markdown("---")
    
    col_g1, col_g2, col_g3 = st.columns(3)
    
    with col_g1:
        st.markdown("**📊 Header Metrics (KPIs)**")
        st.markdown("""
        * **30D Realized Vol (Stock):** Annualized standard deviation of daily log returns over trailing 30 trading days.
        * **30D Realized Vol (S&P 500):** Market baseline volatility for macroeconomic context.
        * **30D GARCH Forecast:** Forward volatility modeled via GARCH(1,1) taking clustering and mean reversion into account.
        * **ATM Implied Vol (IV):** Real-time IV extracted from the front-month call contract nearest spot price.
        * **Vol Premium (IV / HV):** Ratio comparing implied expectations to realized swings.
          * **> 1.0x (Expensive):** Options market prices in elevated future uncertainty relative to past swings (favors net sellers).
          * **< 1.0x (Cheap):** Options market underprices current stock movements (favors net buyers).
        """)

    with col_g2:
        st.markdown("**📈 Primary Charts**")
        st.markdown("""
        * **Historical 30D Realized Volatility:** Plots 30-day rolling HV against the S&P 500 to evaluate regime shifts.
        * **ATM Implied Volatility Logged:** Records live daily ATM IV snapshots to monitor how forward expectations evolve relative to realized volatility.
        * **30-Day Rolling Beta:** Measures systematic sensitivity to market movements ($\beta = \\frac{\\text{Cov}(R_s, R_m)}{\\text{Var}(R_m)}$).
        """)

    with col_g3:
        st.markdown("**🔍 Surface Analytics**")
        st.markdown("""
        * **Volatility Skew / Smile:** Plots put vs. call IV across strikes for the front-month contract. An elevated put curve signals downside crash protection demand.
        * **IV Term Structure:** Displays ATM IV across consecutive expiration dates.
          * **Contango (Upward Sloping):** Standard regime where near-term risk is low relative to long-term uncertainty.
          * **Backwardation (Inverted):** Near-term IV exceeds long-term IV, indicating immediate binary event risk (e.g., earnings releases).
        """)

    st.info("💡 **Quick Start:** Enter any active U.S. equity ticker below to populate the full volatility analysis suite.")

st.markdown("---")

if 'iv_history' not in st.session_state:
    st.session_state.iv_history = pd.DataFrame(columns=['Date', 'Ticker', 'ATM_IV'])

@st.cache_data(ttl=1800)
def fetch_stock_and_index_data(ticker: str, index_ticker: str = "^GSPC"):
    data = yf.download([ticker, index_ticker], period="2y", progress=False)['Close']
    if data.empty or ticker not in data.columns:
        return None
    
    df = pd.DataFrame()
    df['Stock_Close'] = data[ticker]
    df['Index_Close'] = data[index_ticker]
    
    df['Stock_Return'] = np.log(df['Stock_Close'] / df['Stock_Close'].shift(1))
    df['Index_Return'] = np.log(df['Index_Close'] / df['Index_Close'].shift(1))
    df = df.dropna()
    
    df['Stock_30D_Vol'] = df['Stock_Return'].rolling(window=30).std() * np.sqrt(252)
    df['Index_30D_Vol'] = df['Index_Return'].rolling(window=30).std() * np.sqrt(252)
    
    rolling_cov = df['Stock_Return'].rolling(window=30).cov(df['Index_Return'])
    rolling_var = df['Index_Return'].rolling(window=30).var()
    df['Rolling_Beta'] = rolling_cov / rolling_var
    
    return df

@st.cache_data(ttl=900, show_spinner=False)
def fetch_implied_volatility_analytics(ticker: str):
    """Cached call with standard User-Agent headers to prevent Yahoo Finance options rate limits."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    })
    
    for attempt in range(3):
        try:
            tk = yf.Ticker(ticker, session=session)
            expirations = tk.options
            
            if not expirations:
                time.sleep(0.5)
                continue

            fast_info = tk.fast_info
            current_price = fast_info.get('lastPrice') or fast_info.get('previousClose')
            
            if not current_price:
                hist = tk.history(period="5d")
                if hist.empty:
                    return None, None, None, "Unable to retrieve underlying spot price."
                current_price = hist['Close'].iloc[-1]

            atm_data = None
            skew_df = pd.DataFrame()
            
            for exp_date in expirations[:3]:
                try:
                    chain = tk.option_chain(exp_date)
                    calls = chain.calls[chain.calls['impliedVolatility'] > 0.01].copy()
                    puts = chain.puts[chain.puts['impliedVolatility'] > 0.01].copy()
                    
                    if calls.empty:
                        continue
                    
                    calls['strike_diff'] = (calls['strike'] - current_price).abs()
                    atm_contract = calls.sort_values('strike_diff').iloc[0]
                    
                    atm_data = {
                        "iv": float(atm_contract['impliedVolatility']),
                        "expiration": exp_date,
                        "strike": float(atm_contract['strike']),
                        "underlying_price": float(current_price)
                    }
                    
                    calls_skew = calls[(calls['strike'] >= current_price * 0.75) & (calls['strike'] <= current_price * 1.25)][['strike', 'impliedVolatility']].rename(columns={'impliedVolatility': 'Call_IV'})
                    puts_skew = puts[(puts['strike'] >= current_price * 0.75) & (puts['strike'] <= current_price * 1.25)][['strike', 'impliedVolatility']].rename(columns={'impliedVolatility': 'Put_IV'})
                    skew_df = pd.merge(calls_skew, puts_skew, on='strike', how='outer').sort_values('strike')
                    break
                except Exception:
                    continue

            term_structure = []
            for exp_date in expirations[:8]:
                try:
                    c = tk.option_chain(exp_date).calls
                    c_valid = c[c['impliedVolatility'] > 0.01].copy()
                    if not c_valid.empty:
                        c_valid['strike_diff'] = (c_valid['strike'] - current_price).abs()
                        atm_row = c_valid.sort_values('strike_diff').iloc[0]
                        term_structure.append({"Expiration": exp_date, "ATM_IV": float(atm_row['impliedVolatility'])})
                except Exception:
                    continue
                    
            term_df = pd.DataFrame(term_structure)
            return atm_data, skew_df, term_df, None

        except Exception as e:
            if attempt == 2:
                return None, None, None, f"Option analytics error: {str(e)}"
            time.sleep(0.5)

    return None, None, None, "Yahoo Finance rate-limited option chain retrieval. Try clearing cache or re-entering ticker."

def fit_garch(returns: pd.Series, horizon: int = 30) -> float:
    scaled_returns = returns * 100
    model = arch_model(scaled_returns, vol='GARCH', p=1, q=1, mean='constant', dist='normal')
    model_fit = model.fit(disp='off')
    forecast = model_fit.forecast(horizon=horizon)
    avg_daily_var = forecast.variance.iloc[-1].mean()
    return float((np.sqrt(avg_daily_var) / 100) * np.sqrt(252))

# --- Dashboard Input Controls ---
col_input, col_bench = st.columns([2, 2])
with col_input:
    ticker_input = st.text_input("Enter Equity Ticker:", value="NOW").upper().strip()
with col_bench:
    st.text_input("Benchmark Index:", value="^GSPC (S&P 500)", disabled=True)

if ticker_input:
    with st.spinner(f"Fetching market data and calculating volatility analytics for {ticker_input}..."):
        df = fetch_stock_and_index_data(ticker_input)
        iv_data, skew_df, term_df, iv_error = fetch_implied_volatility_analytics(ticker_input)
        
        if df is None or len(df) < 30:
            st.error(f"Could not load sufficient market data for ticker '{ticker_input}'.")
        else:
            stock_30d_hv = df['Stock_30D_Vol'].iloc[-1]
            index_30d_hv = df['Index_30D_Vol'].iloc[-1]
            garch_30 = fit_garch(df['Stock_Return'])

            today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
            if iv_data:
                new_entry = pd.DataFrame([{'Date': today_str, 'Ticker': ticker_input, 'ATM_IV': iv_data['iv']}])
                st.session_state.iv_history = pd.concat([st.session_state.iv_history, new_entry]).drop_duplicates(subset=['Date', 'Ticker'], keep='last')

            # Metric Display Row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("30D Realized Vol (Stock)", f"{stock_30d_hv:.2%}")
            m2.metric("30D Realized Vol (S&P 500)", f"{index_30d_hv:.2%}")
            m3.metric("30D GARCH Forecast", f"{garch_30:.2%}")
            
            if iv_data:
                m4.metric("ATM Implied Vol (IV)", f"{iv_data['iv']:.2%}", help=f"ATM Strike ${iv_data['strike']} (Exp: {iv_data['expiration']})")
                vrp_ratio = iv_data['iv'] / stock_30d_hv if stock_30d_hv > 0 else 0
                m5.metric("Vol Premium (IV / HV)", f"{vrp_ratio:.2f}x", delta="Expensive Options" if vrp_ratio > 1.0 else "Cheap Options", delta_color="normal" if vrp_ratio > 1.0 else "inverse")
            else:
                m4.metric("ATM Implied Vol (IV)", "N/A", delta=iv_error, delta_color="off")
                m5.metric("Vol Premium (IV / HV)", "N/A")

            st.markdown("---")

            # Row 1: Realized Vol & Historical IV Tracker
            r1_col1, r1_col2 = st.columns(2)
            with r1_col1:
                st.subheader("Historical 30-Day Realized Volatility")
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(x=df.index, y=df['Stock_30D_Vol'], mode='lines', name=f'{ticker_input} 30D HV', line=dict(width=2)))
                fig_vol.add_trace(go.Scatter(x=df.index, y=df['Index_30D_Vol'], mode='lines', name='S&P 500 30D HV', line=dict(dash='dash', color='gray')))
                fig_vol.update_layout(xaxis_title="Date", yaxis_title="Annualized Vol", yaxis_tickformat='.0%', template="plotly_white", height=350)
                st.plotly_chart(fig_vol, use_container_width=True)

            with r1_col2:
                st.subheader("ATM Implied Volatility Logged Over Time")
                ticker_iv_hist = st.session_state.iv_history[st.session_state.iv_history['Ticker'] == ticker_input]
                fig_iv = go.Figure()
                fig_iv.add_trace(go.Scatter(x=df.index, y=df['Stock_30D_Vol'], mode='lines', name='30D Realized HV (Baseline)', line=dict(color='lightblue', width=1.5)))
                if not ticker_iv_hist.empty:
                    fig_iv.add_trace(go.Scatter(x=pd.to_datetime(ticker_iv_hist['Date']), y=ticker_iv_hist['ATM_IV'], mode='lines+markers', name='Logged ATM IV', line=dict(color='orange', width=2.5), marker=dict(size=6)))
                fig_iv.update_layout(xaxis_title="Date", yaxis_title="Annualized Vol", yaxis_tickformat='.0%', template="plotly_white", height=350)
                st.plotly_chart(fig_iv, use_container_width=True)

            # Row 2: Advanced Volatility Surface Models
            r2_col1, r2_col2 = st.columns(2)
            with r2_col1:
                st.subheader("Volatility Skew / Smile (Front-Month)")
                if skew_df is not None and not skew_df.empty:
                    fig_skew = go.Figure()
                    fig_skew.add_trace(go.Scatter(x=skew_df['strike'], y=skew_df['Put_IV'], mode='lines+markers', name='Put IV (Downside Skew)', line=dict(color='red')))
                    fig_skew.add_trace(go.Scatter(x=skew_df['strike'], y=skew_df['Call_IV'], mode='lines+markers', name='Call IV (Upside Skew)', line=dict(color='green')))
                    if iv_data:
                        fig_skew.add_vline(x=iv_data['underlying_price'], line_dash="dash", line_color="black", annotation_text=f"Spot ${iv_data['underlying_price']:.2f}")
                    fig_skew.update_layout(xaxis_title="Strike Price ($)", yaxis_title="Implied Volatility", yaxis_tickformat='.0%', template="plotly_white", height=350)
                    st.plotly_chart(fig_skew, use_container_width=True)
                else:
                    st.info("Volatility skew data unavailable for this ticker.")

            with r2_col2:
                st.subheader("IV Term Structure (Across Expirations)")
                if term_df is not None and not term_df.empty:
                    fig_term = go.Figure()
                    fig_term.add_trace(go.Scatter(x=term_df['Expiration'], y=term_df['ATM_IV'], mode='lines+markers', name='ATM IV Term Curve', line=dict(color='purple', width=2.5)))
                    fig_term.update_layout(xaxis_title="Option Expiration Date", yaxis_title="ATM Implied Volatility", yaxis_tickformat='.0%', template="plotly_white", height=350)
                    st.plotly_chart(fig_term, use_container_width=True)
                else:
                    st.info("Term structure data unavailable for this ticker.")

            # Row 3: Rolling Beta
            st.subheader(f"30-Day Rolling Beta ({ticker_input} vs. S&P 500)")
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(x=df.index, y=df['Rolling_Beta'], mode='lines', name='Beta', line=dict(color='teal')))
            fig_beta.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Market Beta (1.0)")
            fig_beta.update_layout(xaxis_title="Date", yaxis_title="Beta Coefficient", template="plotly_white", height=300)
            st.plotly_chart(fig_beta, use_container_width=True)
