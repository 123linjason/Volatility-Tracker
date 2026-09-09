import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="CAN SLIM Equity Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-title {
        color: #8f929d;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 4px;
        white-space: nowrap;
    }
    .metric-value {
        color: #ffffff;
        font-size: 20px;
        font-weight: 700;
        word-wrap: break-word;
        white-space: normal;
    }
    .metric-sub {
        font-size: 13px;
        margin-top: 4px;
    }
    .text-green { color: #089981; }
    .text-red { color: #f23645; }
    .sidebar-expander-header {
        font-weight: 600;
        color: #2962ff;
    }
</style>
""", unsafe_allow_html=True)

def format_large_number(num):
    if num is None or np.isnan(num):
        return "N/A"
    if num >= 1e12:
        return f"${num/1e12:,.2f}T"
    if num >= 1e9:
        return f"${num/1e9:,.2f}B"
    if num >= 1e6:
        return f"${num/1e6:,.2f}M"
    return f"${num:,.2f}"

# ==========================================
# DATA RETRIEVAL ENGINE
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financial_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # 1. Price History
        df_price = ticker.history(period="max", interval="1d")
        if df_price.empty:
            return None, "No price data found for ticker."
        
        if df_price.index.tz is not None:
            df_price.index = df_price.index.tz_localize(None)

        # Benchmark Data (S&P 500)
        sp500 = yf.Ticker("^GSPC").history(period="max", interval="1d")
        if sp500.index.tz is not None:
            sp500.index = sp500.index.tz_localize(None)
            
        df_price = df_price.join(sp500['Close'].rename('SP500_Close'), how='left')
        df_price['SP500_Close'] = df_price['SP500_Close'].ffill().bfill()
        
        # 2. Financial Statements (Fetch up to 12 quarters)
        q_financials = ticker.quarterly_financials
        q_income = ticker.quarterly_incomestmt
        a_financials = ticker.financials
        a_balance = ticker.balance_sheet

        q_combined = q_financials if not q_financials.empty else q_income
        info = ticker.info if ticker.info else {}

        return {
            "price_data": df_price,
            "q_financials": q_combined,
            "a_financials": a_financials,
            "a_balance": a_balance,
            "info": info
        }, None
        
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_peer_benchmark(main_ticker, peer_list):
    """Fetches benchmark comparison metrics for main ticker vs peer list."""
    all_tickers = [main_ticker] + peer_list
    comparison_data = []
    
    for symbol in all_tickers:
        try:
            t = yf.Ticker(symbol)
            info = t.info if t.info else {}
            hist = t.history(period="1y")
            
            # Calculate 1-Year Stock Return
            if not hist.empty and len(hist) > 200:
                ytd_return = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
            else:
                ytd_return = np.nan
                
            comparison_data.append({
                "Ticker": symbol,
                "Company Name": info.get('shortName', symbol),
                "1Y Price Return": f"{ytd_return:+.2f}%" if not np.isnan(ytd_return) else "N/A",
                "Revenue Growth (YoY)": f"{info.get('revenueGrowth', 0)*100:+.2f}%" if info.get('revenueGrowth') else "N/A",
                "Profit Margin": f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else "N/A",
                "ROE": f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else "N/A",
                "Operating Margin": f"{info.get('operatingMargins', 0)*100:.2f}%" if info.get('operatingMargins') else "N/A",
            })
        except Exception:
            continue
            
    return pd.DataFrame(comparison_data)

# ==========================================
# ANALYTICS & FUNDAMENTALS PROCESSING
# ==========================================
def calculate_technicals(df):
    df = df.copy()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Vol_SMA_50'] = df['Volume'].rolling(window=50).mean()
    df['Volume_Surge'] = (df['Volume'] >= 1.45 * df['Vol_SMA_50'])
    
    df['Price_Change'] = df['Close'] - df['Close'].shift(1)
    df['Volume_Color'] = np.where(df['Price_Change'] >= 0, '#089981', '#f23645')
    
    stock_perf = df['Close'] / df['Close'].iloc[0]
    sp_perf = df['SP500_Close'] / df['SP500_Close'].iloc[0]
    df['RS_Line'] = (stock_perf / sp_perf) * 100
    
    return df

def process_quarterly_fundamentals_12q(q_df):
    """Processes quarterly financials up to 12 quarters and identifies EPS acceleration points."""
    if q_df is None or q_df.empty:
        return pd.DataFrame(), None
    
    df = q_df.T.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    eps_col = [c for c in df.columns if 'Diluted EPS' in str(c) or 'Basic EPS' in str(c) or 'Net Income' in str(c)]
    rev_col = [c for c in df.columns if 'Total Revenue' in str(c) or 'Revenue' in str(c)]
    
    col_eps = eps_col[0] if eps_col else None
    col_rev = rev_col[0] if rev_col else None

    summary = pd.DataFrame(index=df.index)
    
    if col_eps:
        summary['Quarterly EPS ($)'] = pd.to_numeric(df[col_eps], errors='coerce')
        summary['EPS YoY Growth (%)'] = summary['Quarterly EPS ($)'].pct_change(4) * 100
    else:
        summary['Quarterly EPS ($)'] = np.nan
        summary['EPS YoY Growth (%)'] = np.nan

    if col_rev:
        summary['Quarterly Revenue ($)'] = pd.to_numeric(df[col_rev], errors='coerce')
        summary['Revenue YoY Growth (%)'] = summary['Quarterly Revenue ($)'].pct_change(4) * 100
    else:
        summary['Quarterly Revenue ($)'] = np.nan
        summary['Revenue YoY Growth (%)'] = np.nan

    # Identify Acceleration Trajectory
    summary['EPS_Accelerating'] = summary['EPS YoY Growth (%)'] > summary['EPS YoY Growth (%)'].shift(1)
    
    # Flag explicit turning point quarter where acceleration started
    summary['Acceleration_Start'] = (summary['EPS_Accelerating']) & (~summary['EPS_Accelerating'].shift(1).fillna(False))
    
    # Limit to latest 12 quarters
    summary = summary.tail(12)
    
    # Find latest acceleration start quarter name
    accel_quarters = summary[summary['Acceleration_Start']].index
    latest_accel_q = accel_quarters[-1].strftime('%B %Y') if len(accel_quarters) > 0 else "N/A"

    # Display formatting
    summary_display = summary.copy()
    summary_display['Status Indicator'] = np.where(
        summary_display['Acceleration_Start'], "🚀 Acceleration Started",
        np.where(summary_display['EPS_Accelerating'], "📈 Accelerating", "🔽 Decelerating")
    )
    
    return summary_display.sort_index(ascending=False), latest_accel_q

def calculate_accurate_roe(info, a_financials, a_balance):
    info_roe = info.get('returnOnEquity')
    if info_roe is not None and not np.isnan(info_roe) and info_roe != 0:
        return info_roe * 100

    try:
        if not a_financials.empty and not a_balance.empty:
            net_income_col = [c for c in a_financials.index if 'Net Income' in str(c)]
            equity_col = [c for c in a_balance.index if 'Stockholders Equity' in str(c) or 'Total Equity' in str(c)]
            if net_income_col and equity_col:
                net_income = a_financials.loc[net_income_col[0]].iloc[0]
                equity = a_balance.loc[equity_col[0]].iloc[0]
                if equity != 0:
                    return (net_income / equity) * 100
    except Exception:
        pass
        
    return None

def detect_chart_patterns(df):
    if len(df) < 200:
        return {"Pattern": "Insufficient Data", "Breakout": False}
    
    recent_df = df.tail(150)
    highs = recent_df['High'].values
    lows = recent_df['Low'].values
    
    peaks, _ = find_peaks(highs, distance=20)
    troughs, _ = find_peaks(-lows, distance=20)
    
    pattern_detected = "Consolidation / Base"
    
    if len(peaks) >= 2 and len(troughs) >= 1:
        left_rim = highs[peaks[0]]
        bottom = lows[troughs[0]]
        right_rim = highs[peaks[-1]]
        depth = (left_rim - bottom) / left_rim
        if 0.12 <= depth <= 0.40 and abs(left_rim - right_rim) / left_rim <= 0.15:
            pattern_detected = "Cup with Handle Pattern"
            
    max_52w = df['High'].tail(252).max()
    latest_close = df['Close'].iloc[-1]
    latest_vol_surge = df['Volume_Surge'].iloc[-1]
    above_10w = latest_close > df['SMA_50'].iloc[-1]
    
    return {
        "Pattern": pattern_detected,
        "Breakout": (latest_close >= 0.95 * max_52w) and latest_vol_surge and above_10w,
        "Near_52W_High": latest_close >= 0.95 * max_52w,
        "Above_10W_SMA": above_10w
    }

# ==========================================
# APPLICATION DASHBOARD & SIDEBAR
# ==========================================
st.title("📈 CAN SLIM Equity Analytics Dashboard")
st.caption("Quantitative screening based on William O'Neil's *How to Make Money in Stocks* principles.")

# Sidebar Configuration & Peer Input
st.sidebar.header("User Settings")
ticker_input = st.sidebar.text_input("Enter Stock Ticker", value="NVDA").upper().strip()
peer_input = st.sidebar.text_input("Enter Key Peer Tickers (Comma-Separated)", value="AMD, AVGO, INTC, TSM").upper().replace(" ", "").split(",")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 CAN SLIM Criterion Breakdown")

if ticker_input:
    with st.spinner(f"Loading analytics for {ticker_input}..."):
        data, err = fetch_financial_data(ticker_input)

    if err or data is None:
        st.error(f"Error fetching data for '{ticker_input}': {err}")
    else:
        df_price = calculate_technicals(data['price_data'])
        q_summary, accel_start_q = process_quarterly_fundamentals_12q(data['q_financials'])
        pattern_info = detect_chart_patterns(df_price)
        info = data['info']
        
        # -------------------------------------------------------------
        # SIDEBAR EXPANDERS (DETAILED CAN SLIM DRAWDOWN)
        # -------------------------------------------------------------
        with st.sidebar.expander("📌 C: Accelerating Quarterly EPS"):
            st.write(f"**Target:** +20% to +40%+ YoY growth.")
            if not q_summary.empty and 'EPS YoY Growth (%)' in q_summary.columns:
                latest_eps = q_summary['EPS YoY Growth (%)'].iloc[0]
                st.write(f"**Latest EPS Growth:** `{latest_eps:+.2f}%`" if not np.isnan(latest_eps) else "N/A")
                st.write(f"**Acceleration Turning Point:** `{accel_start_q}`")
            st.caption("Look for quarters where YoY growth rates systematically increase over prior quarters.")

        with st.sidebar.expander("📌 A: Annual EPS Growth & ROE"):
            calculated_roe = calculate_accurate_roe(info, data['a_financials'], data['a_balance'])
            roe_val = f"{calculated_roe:.2f}%" if calculated_roe is not None else "N/A"
            st.write(f"**Return on Equity (ROE):** `{roe_val}` (Benchmark ≥ 17%)")
            st.write(f"**3-Yr EPS Growth Rate:** `{info.get('earningsQuarterlyGrowth', 0)*100:+.2f}%`")

        with st.sidebar.expander("📌 N: New Products, Highs & Mgmt"):
            st.write(f"**Near 52-Wk High:** {'✅ Yes' if pattern_info['Near_52W_High'] else '❌ No'}")
            st.caption("Companies driving market momentum usually feature innovative products or top-tier leadership changes.")

        with st.sidebar.expander("📌 S: Supply & Volume Demand"):
            vol_surge = df_price['Volume_Surge'].iloc[-1]
            st.write(f"**Volume Spike (>45% Above Avg):** {'✅ Heavy Demand' if vol_surge else '⚪ Normal Volume'}")
            st.write(f"**Float Shares:** `{info.get('floatShares', 0):,}`" if info.get('floatShares') else "Float: N/A")

        with st.sidebar.expander("📌 L: Relative Leader vs Peers"):
            st.write(f"**RS Line Above Benchmark:** {'✅ Strong' if df_price['RS_Line'].iloc[-1] > 100 else '❌ Weak'}")
            st.markdown("**Peer Group Performance Benchmark:**")
            peer_df = fetch_peer_benchmark(ticker_input, peer_input)
            st.dataframe(peer_df, use_container_width=True)

        with st.sidebar.expander("📌 I: Institutional Sponsorship"):
            inst_pct = info.get('heldPercentInstitutions', 0) * 100
            st.write(f"**Institutional Ownership:** `{inst_pct:.2f}%`")
            st.write(f"**Sponsorship Trajectory:** {'✅ High Sponsorship' if inst_pct >= 50 else '⚠️ Low Sponsorship'}")

        with st.sidebar.expander("📌 M: Market Direction Harmony"):
            sp_latest = df_price['SP500_Close'].iloc[-1]
            sp_50d = df_price['SP500_Close'].rolling(50).mean().iloc[-1]
            st.write(f"**S&P 500 Market Trend:** {'🟢 Confirmed Uptrend' if sp_latest > sp_50d else '🔴 Market Correction'}")

        # -------------------------------------------------------------
        # MAIN DASHBOARD METRICS
        # -------------------------------------------------------------
        latest_price = df_price['Close'].iloc[-1]
        prev_price = df_price['Close'].iloc[-2]
        chg = ((latest_price - prev_price) / prev_price) * 100
        chg_class = "text-green" if chg >= 0 else "text-red"
        
        calculated_roe = calculate_accurate_roe(info, data['a_financials'], data['a_balance'])
        roe_display = f"{calculated_roe:.2f}%" if calculated_roe is not None else "N/A"

        min_52w = df_price['Low'].tail(252).min()
        max_52w = df_price['High'].tail(252).max()

        c1, c2, c3, c4, c5 = st.columns(5)
        
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Current Price</div>
                <div class="metric-value">${latest_price:,.2f}</div>
                <div class="metric-sub {chg_class}">{chg:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">52-Week Range</div>
                <div class="metric-value" style="font-size: 16px;">${min_52w:,.2f} - ${max_52w:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Market Cap</div>
                <div class="metric-value">{format_large_number(info.get('marketCap'))}</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ROE</div>
                <div class="metric-value">{roe_display}</div>
            </div>
            """, unsafe_allow_html=True)

        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Base Pattern Detected</div>
                <div class="metric-value" style="font-size: 16px;">{pattern_info['Pattern']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Tabs
        tab_tech, tab_fund, tab_pattern, tab_catalyst = st.tabs([
            "📊 Technicals & Relative Strength",
            "📑 Historical Fundamentals (12 Quarters)",
            "🎯 Pattern Signals & Alerts",
            "🚀 Corporate Catalysts & Peer Benchmarking"
        ])

        # -------------------------------------------------------------
        # TAB 1: TECHNICALS
        # -------------------------------------------------------------
        with tab_tech:
            st.subheader("Price History, Moving Averages & Volume Demand Spikes")
            chart_range = st.radio("Chart Timeframe Filter", ["1 Year", "5 Years", "Max History"], index=0, horizontal=True)
            
            if chart_range == "1 Year":
                plot_df = df_price.tail(252)
                initial_days = 126
            elif chart_range == "5 Years":
                plot_df = df_price.tail(252 * 5)
                initial_days = 252
            else:
                plot_df = df_price
                initial_days = 504

            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.04, 
                subplot_titles=(f"{ticker_input} Candlestick Price & Moving Averages", "Volume (Green = Buying Pressure / Red = Selling Pressure)", "Relative Strength Line vs S&P 500"),
                row_heights=[0.55, 0.22, 0.23]
            )

            fig.add_trace(go.Candlestick(
                x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                low=plot_df['Low'], close=plot_df['Close'], name="Price",
                increasing_line_color='#089981', decreasing_line_color='#f23645'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_50'], name="10-Week SMA", line=dict(color='#2962ff', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_200'], name="40-Week SMA", line=dict(color='#ff6d00', width=1.5)), row=1, col=1)

            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], name="Volume Pressure", marker_color=plot_df['Volume_Color'], opacity=0.8), row=2, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Vol_SMA_50'], name="50-Day Vol Avg", line=dict(color='#f7a900', width=1.5)), row=2, col=1)

            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RS_Line'], name="RS Line vs S&P 500", line=dict(color='#9c27b0', width=2)), row=3, col=1)

            start_date = plot_df.index[-min(initial_days, len(plot_df))]
            end_date = plot_df.index[-1]

            fig.update_layout(
                height=850,
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(range=[start_date, end_date])
            )
            st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------------
        # TAB 2: 12-QUARTER FUNDAMENTALS & EPS ACCELERATION
        # -------------------------------------------------------------
        with tab_fund:
            st.subheader("12-Quarter Revenue & EPS YoY Trajectory")
            
            if not q_summary.empty:
                col_f1, col_f2 = st.columns([2, 1])
                
                with col_f1:
                    fig_fund = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Chronological Plotting
                    q_plot = q_summary.sort_index(ascending=True)
                    
                    fig_fund.add_trace(go.Bar(
                        x=q_plot.index.strftime('%b %Y'), y=q_plot['Quarterly Revenue ($)'], name="Quarterly Revenue ($)", marker_color='#2962ff'
                    ), secondary_y=False)
                    
                    fig_fund.add_trace(go.Scatter(
                        x=q_plot.index.strftime('%b %Y'), y=q_plot['EPS YoY Growth (%)'], name="EPS YoY Growth %",
                        line=dict(color='#089981', width=3), mode='lines+markers'
                    ), secondary_y=True)
                    
                    fig_fund.update_layout(title_text="12-Quarter Sales & EPS Growth Trajectory", template="plotly_dark")
                    fig_fund.update_yaxes(title_text="Sales Revenue ($)", secondary_y=False)
                    fig_fund.update_yaxes(title_text="EPS YoY Growth (%)", secondary_y=True)
                    st.plotly_chart(fig_fund, use_container_width=True)

                with col_f2:
                    st.markdown("### 🚀 Acceleration Diagnostics")
                    st.metric("Latest YoY EPS Growth", f"{q_summary['EPS YoY Growth (%)'].iloc[0]:+.2f}%" if not np.isnan(q_summary['EPS YoY Growth (%)'].iloc[0]) else "N/A")
                    st.metric("Acceleration Turning Point", accel_start_q)
                    
                    if accel_start_q != "N/A":
                        st.success(f"**EPS Acceleration Cycle:** Earnings growth started accelerating in **{accel_start_q}**.")
                    else:
                        st.info("No explicit acceleration turning point detected in the recent 12-quarter window.")

                st.markdown("### Detailed 12-Quarter Fundamental History")
                
                # Format numbers cleanly for table display
                display_df = q_summary.copy()
                display_df.index = display_df.index.strftime('%Y-%m-%d')
                display_df['Quarterly Revenue ($)'] = display_df['Quarterly Revenue ($)'].apply(format_large_number)
                display_df['Quarterly EPS ($)'] = display_df['Quarterly EPS ($)'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
                display_df['EPS YoY Growth (%)'] = display_df['EPS YoY Growth (%)'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")
                display_df['Revenue YoY Growth (%)'] = display_df['Revenue YoY Growth (%)'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")
                
                cols_to_show = ['Quarterly Revenue ($)', 'Revenue YoY Growth (%)', 'Quarterly EPS ($)', 'EPS YoY Growth (%)', 'Status Indicator']
                st.dataframe(display_df[cols_to_show], use_container_width=True)
            else:
                st.warning("Quarterly fundamental data is unavailable for this ticker.")

        # -------------------------------------------------------------
        # TAB 3: PATTERN SIGNALS
        # -------------------------------------------------------------
        with tab_pattern:
            st.subheader("Algorithmic Buy / Sell Alerts & Chart Patterns")
            c_b1, c_b2 = st.columns(2)
            
            with c_b1:
                st.markdown("### 🟢 Buy Signal Evaluation")
                buy_1 = pattern_info['Near_52W_High']
                buy_2 = df_price['Volume_Surge'].iloc[-1]
                buy_3 = pattern_info['Above_10W_SMA']
                
                st.write(f"- Near 52-Week High (within 5%): {'✅ Yes' if buy_1 else '❌ No'}")
                st.write(f"- Heavy Volume Demand Spike (≥45% above avg): {'✅ Yes' if buy_2 else '❌ No'}")
                st.write(f"- Trading Above 10-Week (50-Day) SMA: {'✅ Yes' if buy_3 else '❌ No'}")
                
                if buy_1 and buy_2 and buy_3:
                    st.success("🔥 BUY ALERT: Stock meets classic CAN SLIM breakout criteria!")
                else:
                    st.info("Stock is not currently triggering a classic breakout buy signal.")

            with c_b2:
                st.markdown("### 🔴 Risk & Stop-Loss Rule")
                entry_price = st.number_input("Enter Your Entry Purchase Price ($)", value=float(round(latest_price, 2)))
                stop_loss_price = entry_price * 0.92
                st.markdown(f"**Calculated Cut-Loss Trigger Price (-8%):** `${stop_loss_price:,.2f}`")
                
                if latest_price <= stop_loss_price:
                    st.error("🚨 STOP-LOSS ALERT: Current price has fallen 8% or more below entry point! Cut losses quickly.")
                else:
                    st.success("Current price is above the -8% stop-loss threshold.")

        # -------------------------------------------------------------
        # TAB 4: PEER GROUP BENCHMARKING
        # -------------------------------------------------------------
        with tab_catalyst:
            st.subheader("Relative Leader Benchmarking & Peer Group Comparison")
            
            st.markdown(f"### 🏆 {ticker_input} vs Competitor Peer Group")
            peer_df = fetch_peer_benchmark(ticker_input, peer_input)
            st.dataframe(peer_df, use_container_width=True)
            
            st.markdown("---")
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.markdown("### **N** - New Products & Management Summary")
                st.write(f"**Business Overview:** {info.get('longBusinessSummary', 'N/A')}")
                
            with col_c2:
                st.markdown("### **S** & **I** Factors - Supply & Sponsorship")
                st.write(f"- **Floating Shares:** {info.get('floatShares', 0):,}" if info.get('floatShares') else "- Floating Shares: N/A")
                st.write(f"- **Insider Ownership:** {info.get('heldPercentInsiders', 0)*100:.2f}%" if info.get('heldPercentInsiders') else "- Insider Ownership: N/A")
                st.write(f"- **Institutional Ownership:** {info.get('heldPercentInstitutions', 0)*100:.2f}%" if info.get('heldPercentInstitutions') else "- Institutional Ownership: N/A")
                st.write(f"- **Forward P/E Ratio:** {info.get('forwardPE', 'N/A')}")
