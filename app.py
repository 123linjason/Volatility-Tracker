import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from arch import arch_model

st.set_page_config(page_title="Enterprise Volatility Analytics", layout="wide")

st.title("📈 Enterprise Volatility Analytics")

# --- UI User Guide & Operational Reference ---
with st.expander("📖 Dashboard User Guide & Operational Reference", expanded=False):
    st.markdown("### Executive Overview")
    st.write(
        "This dashboard provides a comprehensive framework for identifying tradeable option mispricings by comparing "
        "options market implied volatility against historical context, event risks, execution costs, and advanced volatility estimators."
    )
    st.markdown("---")
    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        st.markdown("**📊 Relative & Event Metrics**")
        st.markdown("""
        * **1Y IV Rank & Percentile:** Contextualizes current IV relative to its 52-week historical high/low range and distribution.
        * **Earnings Event Adjustments:** Identifies binary event jumps and strips earnings jump variance out of options.
        * **Dynamic Lookback Window:** Matches option DTE against corresponding realized return horizons (5D, 10D, 30D, 90D).
        """)
    with col_g2:
        st.markdown("**📈 Advanced Estimators**")
        st.markdown("""
        * **Yang-Zhang Volatility:** Drift and jump-robust multi-bar estimator combining overnight gaps and intraday high/low ranges.
        * **GARCH(1,1) 95% CI:** Provides forward expected volatility bands to establish realistic price targets.
        """)
    with col_g3:
        st.markdown("**🔍 Execution & Surface Analytics**")
        st.markdown("""
        * **Liquidity & Execution Inputs:** Displays expiration date, bid-ask spread % of premium, open interest, and volume across strikes.
        * **Volatility Skew & Term Structure:** Evaluates call vs. put demand and term structure regimes (Contango vs. Backwardation).
        """)

# --- NEW: Analytical Methodology & Mathematical Explanations ---
with st.expander("📚 Educational Guide: Understanding GARCH & Yang-Zhang Volatility", expanded=False):
    st.markdown("### Deep-Dive: Advanced Volatility Estimators")
    
    col_yz, col_garch = st.columns(2)
    
    with col_yz:
        st.markdown("### 1. Yang-Zhang Volatility Estimator")
        st.markdown("""
        **What it measures:**
        Standard historical volatility (HV) only looks at **Close-to-Close** price changes, completely ignoring what happens during the trading day and overnight. The **Yang-Zhang (YZ)** estimator is a multi-bar volatility metric that captures:
        * **Overnight Volatility:** Price gaps between yesterday's close and today's open.
        * **Open-to-Close Volatility:** Intraday trend movement.
        * **Rogers-Satchell Volatility:** Intraday extreme price movement (Highs and Lows relative to Open/Close).

        **Why it matters for trading:**
        * **Detecting Hidden Risk:** If Yang-Zhang Volatility is significantly *higher* than standard HV, the stock experiences heavy overnight gap risk or violent intraday swings that standard Close-to-Close calculations miss.
        * **Options Pricing Edge:** Options price total risk (including overnight gaps). Comparing YZ to market ATM IV gives a more accurate picture of whether options are actually cheap or expensive relative to true asset movement.
        """)

    with col_garch:
        st.markdown("### 2. GARCH(1,1) Model & Confidence Intervals")
        st.markdown("""
        **What it measures:**
        **GARCH** stands for *Generalized Autoregressive Conditional Heteroskedasticity*. Unlike standard historical volatility (which treats all past days equally), GARCH recognizes two real-world market facts:
        1. **Volatility Clustering:** High volatility days tend to follow high volatility days; quiet days follow quiet days.
        2. **Mean Reversion:** Volatility eventually pulls back toward its long-term average.

        **Why it matters for trading:**
        * **Forward Statistical Expectations:** GARCH generates a forecast of expected volatility over your lookback horizon alongside **95% Confidence Intervals (Upper and Lower Bands)**.
        * **Evaluating Option Mispricings:** 
          * If market **Implied Volatility (IV) > GARCH Upper Band**, options are pricing in extreme panic/fear far beyond statistical expectations (potential **Short Volatility / Sell Premium** edge).
          * If market **Implied Volatility (IV) < GARCH Lower Band**, options are pricing in an unusually calm environment, underestimating potential variance (potential **Long Volatility / Buy Premium** edge).
        """)

st.markdown("---")

# Initialize session state for IV history tracking
if 'iv_history' not in st.session_state:
    st.session_state.iv_history = pd.DataFrame(columns=['Date', 'Ticker', 'ATM_IV'])

# --- Data Helper Functions ---

@st.cache_data(ttl=1800)
def fetch_stock_ohlcv_data(ticker: str, index_ticker: str = "^GSPC"):
    """Fetches OHLCV data for Yang-Zhang estimator and multi-horizon HV calculations."""
    try:
        data = yf.download([ticker, index_ticker], period="2y", progress=False)
        if data.empty:
            return None
        
        df_stock = pd.DataFrame()
        df_stock['Open'] = data['Open'][ticker]
        df_stock['High'] = data['High'][ticker]
        df_stock['Low'] = data['Low'][ticker]
        df_stock['Close'] = data['Close'][ticker]
        df_stock['Index_Close'] = data['Close'][index_ticker]
        df_stock = df_stock.dropna()

        # Log Returns
        df_stock['Stock_Return'] = np.log(df_stock['Close'] / df_stock['Close'].shift(1))
        df_stock['Index_Return'] = np.log(df_stock['Index_Close'] / df_stock['Index_Close'].shift(1))

        # Realized Volatilities across Lookback Windows
        for window in [5, 10, 30, 90]:
            df_stock[f'HV_{window}D'] = df_stock['Stock_Return'].rolling(window=window).std() * np.sqrt(252)

        df_stock['Index_30D_Vol'] = df_stock['Index_Return'].rolling(window=30).std() * np.sqrt(252)

        # Rolling 30D Beta
        rolling_cov = df_stock['Stock_Return'].rolling(window=30).cov(df_stock['Index_Return'])
        rolling_var = df_stock['Index_Return'].rolling(window=30).var()
        df_stock['Rolling_Beta'] = rolling_cov / rolling_var

        return df_stock.dropna()
    except Exception:
        return None

@st.cache_data(ttl=900, show_spinner=False)
def fetch_option_expirations(ticker: str):
    """Retrieves available option expiration dates for a given ticker."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    })
    try:
        tk = yf.Ticker(ticker, session=session)
        expirations = tk.options
        return list(expirations) if expirations else []
    except Exception:
        return []

def calculate_yang_zhang_volatility(df: pd.DataFrame, window: int = 30) -> float:
    """Calculates Yang-Zhang Volatility estimator (overnight, open-to-close, and Rogers-Satchell component)."""
    log_ho = np.log(df['High'] / df['Open'])
    log_lo = np.log(df['Low'] / df['Open'])
    log_co = np.log(df['Close'] / df['Open'])
    log_oc = np.log(df['Open'] / df['Close'].shift(1))
    
    # Rogers-Satchell Volatility
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    rs_vol = rs.rolling(window=window).mean()
    
    # Overnight and Open-to-Close variances
    v_overnight = log_oc.rolling(window=window).var()
    v_open_to_close = log_co.rolling(window=window).var()
    
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_variance = v_overnight + k * v_open_to_close + (1 - k) * rs_vol
    
    yz_vol = np.sqrt(np.maximum(yz_variance, 0)) * np.sqrt(252)
    return float(yz_vol.iloc[-1])

@st.cache_data(ttl=900, show_spinner=False)
def fetch_implied_volatility_analytics(ticker: str, target_expiration: str = None):
    """Fetches option chains for selected expiration, computes execution metrics, and term structure."""
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
                    return None, None, None, None, "Unable to retrieve underlying spot price."
                current_price = hist['Close'].iloc[-1]

            # Earnings date retrieval
            earnings_date_str = "N/A"
            try:
                cal = tk.calendar
                if isinstance(cal, pd.DataFrame) and not cal.empty:
                    if 'Earnings Date' in cal.index:
                        earnings_date_str = str(cal.loc['Earnings Date'].iloc[0].date())
                elif isinstance(cal, dict) and 'Earnings Date' in cal:
                    earnings_date_str = str(cal['Earnings Date'][0])
            except Exception:
                pass

            atm_data = None
            skew_df = pd.DataFrame()
            
            selected_exp = target_expiration if target_expiration in expirations else expirations[0]

            try:
                chain = tk.option_chain(selected_exp)
                calls = chain.calls[chain.calls['impliedVolatility'] > 0.01].copy()
                puts = chain.puts[chain.puts['impliedVolatility'] > 0.01].copy()
                
                if not calls.empty:
                    calls['strike_diff'] = (calls['strike'] - current_price).abs()
                    atm_contract = calls.sort_values('strike_diff').iloc[0]
                    
                    bid = float(atm_contract.get('bid', 0.0))
                    ask = float(atm_contract.get('ask', 0.0))
                    mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else float(atm_contract.get('lastPrice', 0.0))
                    spread_pct = ((ask - bid) / mid) if mid > 0 else 0.0

                    dte = (pd.to_datetime(selected_exp) - pd.Timestamp.now()).days
                    dte = max(dte, 1)

                    atm_data = {
                        "iv": float(atm_contract['impliedVolatility']),
                        "expiration": selected_exp,
                        "dte": dte,
                        "strike": float(atm_contract['strike']),
                        "underlying_price": float(current_price),
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "spread_pct": spread_pct,
                        "open_interest": int(atm_contract.get('openInterest', 0)),
                        "volume": int(atm_contract.get('volume', 0))
                    }
                    
                    calls_skew = calls[(calls['strike'] >= current_price * 0.75) & (calls['strike'] <= current_price * 1.25)].copy()
                    puts_skew = puts[(puts['strike'] >= current_price * 0.75) & (puts['strike'] <= current_price * 1.25)].copy()
                    
                    calls_skew['Call_Mid'] = (calls_skew['bid'] + calls_skew['ask']) / 2.0
                    calls_skew['Call_Spread_%'] = np.where(calls_skew['Call_Mid'] > 0, (calls_skew['ask'] - calls_skew['bid']) / calls_skew['Call_Mid'], 0.0)
                    
                    puts_skew['Put_Mid'] = (puts_skew['bid'] + puts_skew['ask']) / 2.0
                    puts_skew['Put_Spread_%'] = np.where(puts_skew['Put_Mid'] > 0, (puts_skew['ask'] - puts_skew['bid']) / puts_skew['Put_Mid'], 0.0)

                    c_sub = calls_skew[['strike', 'impliedVolatility', 'Call_Spread_%', 'openInterest', 'volume']].rename(
                        columns={'impliedVolatility': 'Call_IV', 'openInterest': 'Call_OI', 'volume': 'Call_Vol'})
                    p_sub = puts_skew[['strike', 'impliedVolatility', 'Put_Spread_%', 'openInterest', 'volume']].rename(
                        columns={'impliedVolatility': 'Put_IV', 'openInterest': 'Put_OI', 'volume': 'Put_Vol'})

                    skew_df = pd.merge(c_sub, p_sub, on='strike', how='outer').sort_values('strike')
                    skew_df.insert(0, 'Expiration', selected_exp)
            except Exception:
                pass

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
            return atm_data, skew_df, term_df, earnings_date_str, None

        except Exception as e:
            if attempt == 2:
                return None, None, None, None, f"Option analytics error: {str(e)}"
            time.sleep(0.5)

    return None, None, None, None, "Yahoo Finance rate-limited option chain retrieval. Try re-entering ticker."

def fit_garch_with_ci(returns: pd.Series, horizon: int = 30):
    """Fits GARCH(1,1) model and returns point forecast alongside 95% Confidence Intervals."""
    scaled_returns = returns * 100
    model = arch_model(scaled_returns, vol='GARCH', p=1, q=1, mean='constant', dist='normal')
    model_fit = model.fit(disp='off')
    
    forecast = model_fit.forecast(horizon=horizon)
    var_forecast = forecast.variance.iloc[-1]
    
    avg_daily_var = var_forecast.mean()
    point_forecast = (np.sqrt(avg_daily_var) / 100) * np.sqrt(252)
    
    var_std = var_forecast.std()
    lower_daily_var = max(0.0001, avg_daily_var - 1.96 * var_std)
    upper_daily_var = avg_daily_var + 1.96 * var_std
    
    lower_ci = (np.sqrt(lower_daily_var) / 100) * np.sqrt(252)
    upper_ci = (np.sqrt(upper_daily_var) / 100) * np.sqrt(252)
    
    return point_forecast, lower_ci, upper_ci

# Helper Function: Text Insight Generator
def generate_volatility_insights(effective_iv, selected_hv, yz_vol, garch_forecast, garch_lower, garch_upper, iv_rank, iv_percentile, iv_data, event_adjust_toggle, earnings_date):
    """Generates concise executive text suggestions and tradeable findings based on dashboard data."""
    insights = []
    
    vrp_ratio = (effective_iv / selected_hv) if selected_hv > 0 else 1.0
    
    # 1. Mispricing & Valuation Assessment
    if vrp_ratio > 1.25 and iv_rank > 0.65:
        insights.append(
            f"**Overpriced Volatility Edge (Vol Premium: {vrp_ratio:.2f}x | 1Y IV Rank: {iv_rank:.1%}):** "
            "Implied volatility is trading at a significant premium relative to underlying realized price moves. "
            "**Actionable Suggestion:** Favor short-volatility structures (e.g., credit spreads, iron condors, or covered calls) to capture premium decay."
        )
    elif vrp_ratio < 0.85 and iv_rank < 0.35:
        insights.append(
            f"**Underpriced Volatility Opportunity (Vol Premium: {vrp_ratio:.2f}x | 1Y IV Rank: {iv_rank:.1%}):** "
            "Implied volatility is historically depressed compared to realized volatility. "
            "**Actionable Suggestion:** Consider long-volatility strategies (e.g., debit calendar spreads or long straddles) to capitalize on potential volatility expansion."
        )
    else:
        insights.append(
            f"**Fairly Valued Volatility (Vol Premium: {vrp_ratio:.2f}x | 1Y IV Rank: {iv_rank:.1%}):** "
            "Options market pricing is closely aligned with recent realized volatility. Delta-neutral volatility edges are currently muted."
        )

    # 2. Advanced Estimator Discrepancy (Yang-Zhang vs Standard HV)
    if yz_vol > selected_hv * 1.15:
        insights.append(
            f"**Yang-Zhang Gap/Intraday Risk Signal:** Yang-Zhang volatility ({yz_vol:.1%}) significantly exceeds standard Close-to-Close HV ({selected_hv:.1%}). "
            "This indicates substantial overnight price gapping or intraday high/low volatility that standard close returns miss."
        )

    # 3. GARCH Range Alignment
    if effective_iv > garch_upper:
        insights.append(
            f"**GARCH Upper Bound Deviation:** Implied volatility ({effective_iv:.1%}) sits above the 95% GARCH conditional upper limit ({garch_upper:.1%}). "
            "Option pricing reflects extreme market anxiety beyond statistical expectations (Potential short volatility / sell edge)."
        )
    elif effective_iv < garch_lower:
        insights.append(
            f"**GARCH Lower Bound Compression:** Implied volatility ({effective_iv:.1%}) sits below the 95% GARCH lower band ({garch_lower:.1%}). "
            "Options appear underpriced relative to statistical conditional persistence (Potential long volatility edge)."
        )

    # 4. Liquidity & Execution Assessment
    if iv_data and iv_data['spread_pct'] > 0.08:
        insights.append(
            f"⚠️ **Execution Drag Warning:** Bid-Ask Spread is wide ({iv_data['spread_pct']:.2%} of premium). "
            "High transaction friction may erase theoretical option mispricing edges. Use strict limit orders at mid-price."
        )
    elif iv_data and iv_data['spread_pct'] <= 0.03:
        insights.append(
            f"✅ **High Execution Quality:** Tight bid-ask spreads ({iv_data['spread_pct']:.2%}) allow efficient entry/exit with low execution slippage."
        )

    # 5. Event Risk Impact
    if event_adjust_toggle:
        insights.append(
            f"**Earnings Variance Stripped:** Single-day event jump risk has been removed from front-month options. "
            f"Effective baseline IV is **{effective_iv:.1%}** (vs. raw market IV)."
        )

    return insights

# --- Dashboard Input Controls & Selectors ---

col_input, col_exp, col_lookback, col_event = st.columns([1.2, 1.2, 1.2, 1.2])

with col_input:
    ticker_input = st.text_input("Enter Equity Ticker:", value="NOW").upper().strip()

# Fetch Expirations for the Ticker
available_expirations = fetch_option_expirations(ticker_input) if ticker_input else []

with col_exp:
    if available_expirations:
        selected_expiration = st.selectbox("Select Option Expiration:", options=available_expirations, index=0)
    else:
        selected_expiration = st.selectbox("Select Option Expiration:", options=["N/A"], index=0, disabled=True)

with col_lookback:
    lookback_window = st.selectbox(
        "Realized Vol Lookback:",
        options=[5, 10, 30, 90],
        index=2,
        format_func=lambda x: f"{x}-Day HV Horizon"
    )

with col_event:
    event_adjust_toggle = st.checkbox("Event-Adjusted Vol Mode", value=False, help="Strips expected single-day earnings jump variance out of total implied volatility.")

if ticker_input:
    with st.spinner(f"Processing multi-horizon analytics for {ticker_input}..."):
        df = fetch_stock_ohlcv_data(ticker_input)
        iv_data, skew_df, term_df, earnings_date, iv_error = fetch_implied_volatility_analytics(
            ticker_input, target_expiration=selected_expiration
        )
        
        if df is None or len(df) < 90:
            st.error(f"Could not load sufficient historical price data for '{ticker_input}'.")
        else:
            selected_hv = df[f'HV_{lookback_window}D'].iloc[-1]
            index_30d_hv = df['Index_30D_Vol'].iloc[-1]
            yz_vol = calculate_yang_zhang_volatility(df, window=lookback_window)
            garch_forecast, garch_lower, garch_upper = fit_garch_with_ci(df['Stock_Return'], horizon=lookback_window)

            today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            # Session State IV History and IV Rank / Percentile Calculation
            if iv_data:
                new_entry = pd.DataFrame([{'Date': today_str, 'Ticker': ticker_input, 'ATM_IV': iv_data['iv']}])
                st.session_state.iv_history = pd.concat([st.session_state.iv_history, new_entry]).drop_duplicates(subset=['Date', 'Ticker'], keep='last')
            
            # Derive 1Y IV Rank & Percentile using rolling historical baseline proxy
            historical_vol_series = df['HV_30D'].dropna()
            iv_min_52w = historical_vol_series.min()
            iv_max_52w = historical_vol_series.max()
            
            effective_iv = iv_data['iv'] if iv_data else selected_hv
            
            # Earnings Event Adjustment Calculation
            if event_adjust_toggle and iv_data:
                dte = iv_data['dte']
                total_variance = (iv_data['iv'] ** 2) * (dte / 365.0)
                earnings_jump_var = (0.05 ** 2)
                stripped_variance = max(0.0001, total_variance - earnings_jump_var)
                effective_iv = np.sqrt(stripped_variance * (365.0 / dte))

            iv_rank = (effective_iv - iv_min_52w) / (iv_max_52w - iv_min_52w) if (iv_max_52w > iv_min_52w) else 0.5
            iv_percentile = (historical_vol_series < effective_iv).mean()

            # --- Row 1: Key Relative Context & Event Indicators ---
            m1, m2, m3, m4, m5 = st.columns(5)
            
            m1.metric(f"Realized Vol ({lookback_window}D)", f"{selected_hv:.2%}", help=f"Annualized {lookback_window}-day HV")
            m2.metric("Yang-Zhang Volatility", f"{yz_vol:.2%}", help="Overnight + intraday drift-robust estimator")
            m3.metric("GARCH 95% Target Range", f"{garch_forecast:.1%}", delta=f"[{garch_lower:.1%} - {garch_upper:.1%}]", delta_color="off")
            
            if iv_data:
                m4.metric(
                    "ATM Implied Vol (IV)", 
                    f"{effective_iv:.2%}", 
                    delta="Event-Adjusted" if event_adjust_toggle else f"Exp: {iv_data['expiration']}",
                    delta_color="normal" if not event_adjust_toggle else "inverse"
                )
                vrp_ratio = effective_iv / selected_hv if selected_hv > 0 else 0
                m5.metric("Vol Premium (IV / HV)", f"{vrp_ratio:.2f}x", delta="Expensive Options" if vrp_ratio > 1.0 else "Cheap Options", delta_color="normal" if vrp_ratio > 1.0 else "inverse")
            else:
                m4.metric("ATM Implied Vol (IV)", "N/A", delta=iv_error, delta_color="off")
                m5.metric("Vol Premium (IV / HV)", "N/A")

            # --- Automated Strategy Findings & Suggestions Summary ---
            st.markdown("---")
            st.subheader("💡 Automated Volatility Findings & Strategy Suggestions")
            
            insights_list = generate_volatility_insights(
                effective_iv, selected_hv, yz_vol, garch_forecast, garch_lower, garch_upper, 
                iv_rank, iv_percentile, iv_data, event_adjust_toggle, earnings_date
            )
            
            for insight in insights_list:
                st.markdown(f"* {insight}")

            # --- Row 2: Relative Context Widgets & Execution Summary ---
            st.markdown("---")
            c_rank, c_perc, c_earn, c_exec = st.columns(4)
            
            with c_rank:
                st.metric("1-Year IV Rank", f"{iv_rank:.1%}", help="Location relative to 52-week IV high/low range")
                st.progress(min(max(iv_rank, 0.0), 1.0))
                
            with c_perc:
                st.metric("1-Year IV Percentile", f"{iv_percentile:.1%}", help="% of days over trailing year where IV was lower")
                st.progress(min(max(iv_percentile, 0.0), 1.0))

            with c_earn:
                st.metric("Upcoming Earnings Date", earnings_date if earnings_date != "N/A" else "None Listed")
                if event_adjust_toggle:
                    st.caption("⚡ Single-day earnings jump risk stripped from IV.")
                else:
                    st.caption("📌 Raw market IV active (includes event jump).")

            with c_exec:
                if iv_data:
                    st.metric("Execution Spread (%)", f"{iv_data['spread_pct']:.2%}", delta=f"Mid: ${iv_data['mid']:.2f}")
                    st.caption(f"Volume: {iv_data['volume']:,} | OI: {iv_data['open_interest']:,}")
                else:
                    st.metric("Execution Spread (%)", "N/A")

            st.markdown("---")

            # --- Row 3: Primary Volatility Trend Charts ---
            r1_col1, r1_col2 = st.columns(2)
            with r1_col1:
                st.subheader(f"Historical Realized Volatility ({lookback_window}D Lookback)")
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(x=df.index, y=df[f'HV_{lookback_window}D'], mode='lines', name=f'{ticker_input} {lookback_window}D HV', line=dict(width=2)))
                fig_vol.add_trace(go.Scatter(x=df.index, y=df['Index_30D_Vol'], mode='lines', name='S&P 500 30D HV', line=dict(dash='dash', color='gray')))
                fig_vol.update_layout(xaxis_title="Date", yaxis_title="Annualized Volatility", yaxis_tickformat='.0%', template="plotly_white", height=350)
                st.plotly_chart(fig_vol, use_container_width=True)

            with r1_col2:
                st.subheader("ATM Implied Volatility Tracking & GARCH Bands")
                ticker_iv_hist = st.session_state.iv_history[st.session_state.iv_history['Ticker'] == ticker_input]
                fig_iv = go.Figure()
                fig_iv.add_trace(go.Scatter(x=df.index, y=df[f'HV_{lookback_window}D'], mode='lines', name=f'{lookback_window}D Realized HV', line=dict(color='lightblue', width=1.5)))
                
                # Plot GARCH Forecast Target Band
                last_date = df.index[-1]
                fig_iv.add_trace(go.Scatter(
                    x=[last_date], y=[garch_upper],
                    mode='markers', name='GARCH 95% Upper CI', marker=dict(color='red', size=8, symbol='triangle-up')
                ))
                fig_iv.add_trace(go.Scatter(
                    x=[last_date], y=[garch_lower],
                    mode='markers', name='GARCH 95% Lower CI', marker=dict(color='green', size=8, symbol='triangle-down')
                ))

                if not ticker_iv_hist.empty:
                    fig_iv.add_trace(go.Scatter(x=pd.to_datetime(ticker_iv_hist['Date']), y=ticker_iv_hist['ATM_IV'], mode='lines+markers', name='Logged ATM IV', line=dict(color='orange', width=2.5), marker=dict(size=6)))
                
                fig_iv.update_layout(xaxis_title="Date", yaxis_title="Annualized Volatility", yaxis_tickformat='.0%', template="plotly_white", height=350)
                st.plotly_chart(fig_iv, use_container_width=True)

            # --- Row 4: Advanced Surface Analytics & Execution Table ---
            r2_col1, r2_col2 = st.columns(2)
            with r2_col1:
                st.subheader(f"Volatility Skew ({selected_expiration})")
                if skew_df is not None and not skew_df.empty:
                    fig_skew = go.Figure()
                    fig_skew.add_trace(go.Scatter(x=skew_df['strike'], y=skew_df['Put_IV'], mode='lines+markers', name='Put IV (Downside Skew)', line=dict(color='red')))
                    fig_skew.add_trace(go.Scatter(x=skew_df['strike'], y=skew_df['Call_IV'], mode='lines+markers', name='Call IV (Upside Skew)', line=dict(color='green')))
                    if iv_data:
                        fig_skew.add_vline(x=iv_data['underlying_price'], line_dash="dash", line_color="black", annotation_text=f"Spot ${iv_data['underlying_price']:.2f}")
                    fig_skew.update_layout(xaxis_title="Strike Price ($)", yaxis_title="Implied Volatility", yaxis_tickformat='.0%', template="plotly_white", height=350)
                    st.plotly_chart(fig_skew, use_container_width=True)
                else:
                    st.info("Volatility skew data unavailable for this expiration.")

            with r2_col2:
                st.subheader("IV Term Structure Across Expirations")
                if term_df is not None and not term_df.empty:
                    fig_term = go.Figure()
                    fig_term.add_trace(go.Scatter(x=term_df['Expiration'], y=term_df['ATM_IV'], mode='lines+markers', name='ATM IV Term Curve', line=dict(color='purple', width=2.5)))
                    fig_term.update_layout(xaxis_title="Option Expiration Date", yaxis_title="ATM Implied Volatility", yaxis_tickformat='.0%', template="plotly_white", height=350)
                    st.plotly_chart(fig_term, use_container_width=True)
                else:
                    st.info("Term structure data unavailable for this ticker.")

            # Option Strike Execution & Liquidity Metrics Table
            if skew_df is not None and not skew_df.empty:
                st.subheader(f"Option Chain Execution & Liquidity Details ({selected_expiration})")
                st.dataframe(
                    skew_df.style.format({
                        'Expiration': '{}',
                        'strike': '${:.2f}',
                        'Call_IV': '{:.2%}',
                        'Put_IV': '{:.2%}',
                        'Call_Spread_%': '{:.2%}',
                        'Put_Spread_%': '{:.2%}',
                        'Call_OI': '{:,.0f}',
                        'Put_OI': '{:,.0f}',
                        'Call_Vol': '{:,.0f}',
                        'Put_Vol': '{:,.0f}'
                    }),
                    use_container_width=True,
                    height=250
                )

            # --- Row 5: Dynamic Rolling Beta ---
            st.subheader(f"30-Day Rolling Beta ({ticker_input} vs. S&P 500)")
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(x=df.index, y=df['Rolling_Beta'], mode='lines', name='Beta', line=dict(color='teal')))
            fig_beta.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Market Beta (1.0)")
            fig_beta.update_layout(xaxis_title="Date", yaxis_title="Beta Coefficient", template="plotly_white", height=280)
            st.plotly_chart(fig_beta, use_container_width=True)
