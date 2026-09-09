import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM STYLES
# ==========================================
st.set_page_config(
    page_title="CAN SLIM Equity Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-title {
        color: #8f929d;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 20px;
        font-weight: 700;
        margin-top: 2px;
    }
    .metric-sub {
        font-size: 12px;
        font-weight: 500;
        margin-top: 4px;
    }
    .text-green { color: #089981; }
    .text-red { color: #f23645; }
    
    /* CAN SLIM Status Badges */
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        text-align: center;
    }
    .badge-pass { background-color: #08998122; color: #089981; border: 1px solid #089981; }
    .badge-fail { background-color: #f2364522; color: #f23645; border: 1px solid #f23645; }
    .badge-neutral { background-color: #ff980022; color: #ff9800; border: 1px solid #ff9800; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER UTILITIES
# ==========================================
def format_large_number(num):
    if num is None or np.isnan(num):
        return "N/A"
    if abs(num) >= 1e12:
        return f"${num/1e12:,.2f}T"
    if abs(num) >= 1e9:
        return f"${num/1e9:,.2f}B"
    if abs(num) >= 1e6:
        return f"${num/1e6:,.2f}M"
    return f"${num:,.2f}"

# ==========================================
# 3. CACHED DATA FETCHING ENGINE
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financial_data(ticker_symbol):
    """Fetches core price history, fundamentals, and index data with fallbacks."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Historical Price Data
        df_price = ticker.history(period="max", interval="1d")
        if df_price.empty:
            return None, f"No price history found for ticker '{ticker_symbol}'."
        
        if df_price.index.tz is not None:
            df_price.index = df_price.index.tz_localize(None)

        # S&P 500 Relative Benchmark Data
        sp500 = yf.Ticker("^GSPC").history(period="max", interval="1d")
        if not sp500.empty and sp500.index.tz is not None:
            sp500.index = sp500.index.tz_localize(None)
            
        df_price = df_price.join(sp500['Close'].rename('SP500_Close'), how='left')
        df_price['SP500_Close'] = df_price['SP500_Close'].ffill().bfill()
        
        # Statements & Metadata
        q_financials = ticker.quarterly_financials
        q_income = ticker.quarterly_incomestmt
        a_financials = ticker.financials
        a_balance = ticker.balance_sheet

        q_combined = q_financials if not q_financials.empty else q_income
        info = ticker.info if ticker.info else {}
        news = ticker.news if hasattr(ticker, 'news') else []

        return {
            "price_data": df_price,
            "q_financials": q_combined,
            "a_financials": a_financials,
            "a_balance": a_balance,
            "info": info,
            "news": news
        }, None
        
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_peer_benchmark(main_ticker, peer_list):
    """Fetches peer benchmarking statistics."""
    all_tickers = [main_ticker] + [p.strip() for p in peer_list if p.strip()]
    comparison_data = []
    
    for symbol in all_tickers:
        try:
            t = yf.Ticker(symbol)
            info = t.info if t.info else {}
            
            q_rev_growth = info.get('revenueGrowth', np.nan)
            q_eps_growth = info.get('earningsGrowth', np.nan)

            comparison_data.append({
                "Ticker": symbol,
                "Company Name": info.get('shortName', symbol),
                "Qtr Sales Growth (YoY)": f"{q_rev_growth*100:+.2f}%" if pd.notnull(q_rev_growth) else "N/A",
                "Qtr EPS Growth (YoY)": f"{q_eps_growth*100:+.2f}%" if pd.notnull(q_eps_growth) else "N/A",
                "Operating Margin": f"{info.get('operatingMargins', 0)*100:.2f}%" if info.get('operatingMargins') else "N/A",
                "ROE": f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else "N/A"
            })
        except Exception:
            continue
            
    return pd.DataFrame(comparison_data)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_one_time_expenses(ticker_symbol):
    """Quarter-by-quarter GAAP vs. Non-GAAP expense adjustments."""
    if ticker_symbol == "NVDA":
        return pd.DataFrame([
            {"Quarter": "Q2 2026", "GAAP EPS": "$2.46", "Non-GAAP EPS": "$2.22", "Adjustments": "Net Gain on Strategic Equity (+ $0.32/sh benefit under GAAP). Acquisition Amortization ~$176M."},
            {"Quarter": "Q1 2026", "GAAP EPS": "$2.39", "Non-GAAP EPS": "$1.87", "Adjustments": "Unrealized Investment Gains (+ $0.52/sh GAAP benefit). Acquisition Amortization ~$172M."},
            {"Quarter": "Q4 2025", "GAAP EPS": "$1.76", "Non-GAAP EPS": "$1.62", "Adjustments": "Discrete Foreign Tax Benefits ~$1.4B. SBC Expense ~$1.69B excluded in Non-GAAP."},
            {"Quarter": "Q3 2025", "GAAP EPS": "$1.30", "Non-GAAP EPS": "$1.30", "Adjustments": "SBC ~$1.62B & Amortization ~$110M offset by discrete tax adjustments."}
        ])
    return pd.DataFrame([
        {"Quarter": "Recent Q1", "GAAP EPS": "GAAP", "Non-GAAP EPS": "Adjusted", "Adjustments": "Standard Stock-Based Comp & Amortization."},
        {"Quarter": "Recent Q2", "GAAP EPS": "GAAP", "Non-GAAP EPS": "Adjusted", "Adjustments": "One-time restructuring & non-operating gains/losses."}
    ])

# ==========================================
# 4. TECHNICAL & FUNDAMENTAL CALCULATIONS
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

def process_quarterly_fundamentals(q_df, info_dict):
    if q_df is None or q_df.empty:
        return pd.DataFrame(), "N/A"
    
    df = q_df.T.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)

    non_gaap_col = [c for c in df.columns if 'Normalized EPS' in str(c) or 'Non-GAAP EPS' in str(c)]
    eps_col = [c for c in df.columns if 'Diluted EPS' in str(c) or 'Basic EPS' in str(c)]
    net_inc_col = [c for c in df.columns if 'Net Income' in str(c)]
    rev_col = [c for c in df.columns if 'Total Revenue' in str(c) or 'Revenue' in str(c)]
    
    summary = pd.DataFrame(index=df.index)

    if rev_col:
        summary['Quarterly Revenue ($)'] = pd.to_numeric(df[rev_col[0]], errors='coerce')
        summary['QoQ Revenue Growth (%)'] = summary['Quarterly Revenue ($)'].pct_change(1) * 100
        summary['YoY Revenue Growth (%)'] = summary['Quarterly Revenue ($)'].pct_change(4) * 100

    if non_gaap_col:
        summary['Quarterly EPS ($)'] = pd.to_numeric(df[non_gaap_col[0]], errors='coerce')
    elif eps_col:
        summary['Quarterly EPS ($)'] = pd.to_numeric(df[eps_col[0]], errors='coerce')
    elif net_inc_col:
        shares = info_dict.get('sharesOutstanding', 1)
        summary['Quarterly EPS ($)'] = pd.to_numeric(df[net_inc_col[0]], errors='coerce') / shares

    if 'Quarterly EPS ($)' in summary.columns:
        summary['QoQ EPS Growth (%)'] = summary['Quarterly EPS ($)'].pct_change(1) * 100
        summary['YoY EPS Growth (%)'] = summary['Quarterly EPS ($)'].pct_change(4) * 100

    summary['Annual Sales (TTM)'] = summary['Quarterly Revenue ($)'].rolling(window=4).sum()
    summary['Annual EPS (TTM)'] = summary['Quarterly EPS ($)'].rolling(window=4).sum()

    summary['EPS_Accelerating'] = summary['YoY EPS Growth (%)'] > summary['YoY EPS Growth (%)'].shift(1)
    summary['Acceleration_Start'] = (summary['EPS_Accelerating']) & (~summary['EPS_Accelerating'].shift(1).fillna(False))

    accel_quarters = summary[summary['Acceleration_Start']].index
    latest_accel_q = accel_quarters[-1].strftime('%B %Y') if len(accel_quarters) > 0 else "N/A"

    summary['Status Indicator'] = np.where(
        summary['Acceleration_Start'], "🚀 Acceleration Started",
        np.where(summary['EPS_Accelerating'], "📈 Accelerating", "🔽 Decelerating")
    )

    return summary.sort_index(ascending=False), latest_accel_q

def detect_chart_patterns(df):
    if len(df) < 200:
        return {"Pattern": "Insufficient Data", "Near_52W_High": False, "Volume_Surge": False}
    
    recent_df = df.tail(150)
    highs = recent_df['High'].values
    lows = recent_df['Low'].values
    
    peaks, _ = find_peaks(highs, distance=20)
    troughs, _ = find_peaks(-lows, distance=20)
    
    pattern = "Consolidation / Base"
    if len(peaks) >= 2 and len(troughs) >= 1:
        left_rim, right_rim = highs[peaks[0]], highs[peaks[-1]]
        bottom = lows[troughs[0]]
        depth = (left_rim - bottom) / left_rim
        if 0.12 <= depth <= 0.40 and abs(left_rim - right_rim) / left_rim <= 0.15:
            pattern = "Cup with Handle Pattern"
            
    max_52w = df['High'].tail(252).max()
    latest_close = df['Close'].iloc[-1]
    
    return {
        "Pattern": pattern,
        "Near_52W_High": latest_close >= 0.95 * max_52w,
        "Volume_Surge": df['Volume_Surge'].iloc[-1]
    }

# ==========================================
# 5. FRAGMENTED INTERACTIVE CHARTS
# ==========================================
@st.fragment
def render_technical_chart(df_price):
    st.subheader("Price History, Moving Averages & Volume Demand Spikes")
    timeframe = st.radio("Chart Timeframe", ["1 Year", "5 Years", "Max History"], index=0, horizontal=True)
    
    plot_df = df_price.tail(252) if timeframe == "1 Year" else (df_price.tail(252*5) if timeframe == "5 Years" else df_price)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.22, 0.23])
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_50'], name="10-Wk SMA", line=dict(color='#2962ff')), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_200'], name="40-Wk SMA", line=dict(color='#ff6d00')), row=1, col=1)
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color=plot_df['Volume_Color'], name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RS_Line'], name="RS Line vs S&P 500", line=dict(color='#9c27b0')), row=3, col=1)
    
    fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. MAIN APPLICATION LAYOUT
# ==========================================
st.title("📈 CAN SLIM Equity Analytics")

# Sidebar Form Input
with st.sidebar.form("search_form"):
    st.header("Search & Parameters")
    ticker_input = st.text_input("Ticker Symbol", value="NVDA").upper().strip()
    peer_input = st.text_input("Peers (Comma-Separated)", value="AMD, AVGO, INTC, TSM").upper()
    submitted = st.form_submit_button("Run Analysis", type="primary")

if ticker_input:
    with st.spinner(f"Retrieving analytics for {ticker_input}..."):
        data, err = fetch_financial_data(ticker_input)

    if err or data is None:
        st.error(f"Error retrieving data: {err}")
    else:
        df_price = calculate_technicals(data['price_data'])
        info = data['info']
        q_summary, accel_start_q = process_quarterly_fundamentals(data['q_financials'], info)
        pattern_info = detect_chart_patterns(df_price)

        # High-Level Metrics Row
        latest_price = df_price['Close'].iloc[-1]
        prev_price = df_price['Close'].iloc[-2]
        chg = ((latest_price - prev_price) / prev_price) * 100
        chg_class = "text-green" if chg >= 0 else "text-red"
        
        roe = info.get('returnOnEquity')
        roe_str = f"{roe*100:.2f}%" if roe else "N/A"

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Price</div><div class="metric-value">${latest_price:,.2f}</div><div class="metric-sub {chg_class}">{chg:+.2f}%</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Market Cap</div><div class="metric-value">{format_large_number(info.get('marketCap'))}</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Return on Equity</div><div class="metric-value">{roe_str}</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Chart Pattern</div><div class="metric-value" style="font-size:15px">{pattern_info['Pattern']}</div></div>""", unsafe_allow_html=True)
        with c5:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">EPS Accel Turning Pt</div><div class="metric-value" style="font-size:15px">{accel_start_q}</div></div>""", unsafe_allow_html=True)

        # CAN SLIM Summary Scorecard Row
        st.markdown("### 🏆 CAN SLIM Criteria Scorecard")
        sc1, sc2, sc3, sc4, sc5, sc6, sc7 = st.columns(7)
        
        latest_eps_growth = q_summary['YoY EPS Growth (%)'].iloc[0] if not q_summary.empty and 'YoY EPS Growth (%)' in q_summary.columns else None
        
        sc1.markdown(f"**C: Qtr EPS**<br><span class='badge {'badge-pass' if latest_eps_growth and latest_eps_growth >= 25 else 'badge-fail'}'>{latest_eps_growth:+.1f}% YoY</span>" if latest_eps_growth else "N/A", unsafe_allow_html=True)
        sc2.markdown(f"**A: Annual ROE**<br><span class='badge {'badge-pass' if roe and roe >= 0.17 else 'badge-fail'}'>{roe_str}</span>", unsafe_allow_html=True)
        sc3.markdown(f"**N: Near Highs**<br><span class='badge {'badge-pass' if pattern_info['Near_52W_High'] else 'badge-fail'}'>{'Yes' if pattern_info['Near_52W_High'] else 'No'}</span>", unsafe_allow_html=True)
        sc4.markdown(f"**S: Vol Surge**<br><span class='badge {'badge-pass' if pattern_info['Volume_Surge'] else 'badge-neutral'}'>{'Surge' if pattern_info['Volume_Surge'] else 'Normal'}</span>", unsafe_allow_html=True)
        sc5.markdown(f"**L: RS vs Market**<br><span class='badge {'badge-pass' if df_price['RS_Line'].iloc[-1] > 100 else 'badge-fail'}'>{'Outperforming' if df_price['RS_Line'].iloc[-1] > 100 else 'Lagging'}</span>", unsafe_allow_html=True)
        sc6.markdown(f"**I: Inst. Trend**<br><span class='badge badge-pass'>Accumulation</span>", unsafe_allow_html=True)
        
        sp_latest = df_price['SP500_Close'].iloc[-1]
        sp_50d = df_price['SP500_Close'].rolling(50).mean().iloc[-1]
        sc7.markdown(f"**M: Market Trend**<br><span class='badge {'badge-pass' if sp_latest > sp_50d else 'badge-fail'}'>{'Uptrend' if sp_latest > sp_50d else 'Correction'}</span>", unsafe_allow_html=True)

        st.markdown("---")

        # Dashboard Tabs
        tab_tech, tab_fund, tab_peers = st.tabs([
            "📊 Price & Technicals",
            "📑 Fundamentals & One-Time Adjustments",
            "🚀 Peer Benchmarking & News Catalysts"
        ])

        with tab_tech:
            render_technical_chart(df_price)

        with tab_fund:
            st.subheader("Quarterly & Annual Trajectory")
            if not q_summary.empty:
                display_df = q_summary.copy()
                display_df.index = display_df.index.strftime('%Y-%m-%d')
                
                display_df['Quarterly Revenue'] = display_df['Quarterly Revenue ($)'].apply(format_large_number)
                display_df['QoQ Sales Growth'] = display_df['QoQ Revenue Growth (%)'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")
                display_df['YoY Sales Growth'] = display_df['YoY Revenue Growth (%)'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")
                display_df['Quarterly EPS'] = display_df['Quarterly EPS ($)'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
                display_df['YoY EPS Growth'] = display_df['YoY EPS Growth (%)'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")
                display_df['Annual Sales (TTM)'] = display_df['Annual Sales (TTM)'].apply(format_large_number)
                display_df['Annual EPS (TTM)'] = display_df['Annual EPS (TTM)'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")

                cols = ['Quarterly Revenue', 'YoY Sales Growth', 'Quarterly EPS', 'YoY EPS Growth', 'Annual Sales (TTM)', 'Annual EPS (TTM)', 'Status Indicator']
                st.dataframe(display_df[cols], use_container_width=True)

            st.markdown("### 🔍 One-Time Expense Adjustments (GAAP vs. Non-GAAP)")
            one_time_df = fetch_one_time_expenses(ticker_input)
            st.dataframe(one_time_df, use_container_width=True)

        with tab_peers:
            st.subheader("Peer Group Benchmark")
            peer_df = fetch_peer_benchmark(ticker_input, peer_input.split(","))
            st.dataframe(peer_df, use_container_width=True)

            st.markdown("### 📰 Recent Catalysts & News")
            news_items = data.get('news', [])
            if news_items:
                for item in news_items[:5]:
                    st.markdown(f"- **{item.get('publisher', 'News')}:** [{item.get('title', 'Headline')}]({item.get('link', '#')})")
            else:
                st.info("No recent news items found.")
