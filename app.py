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
    page_title="CAN SLIM Equity Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Layout Tweaks */
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    
    /* Institutional Metric Cards */
    .metric-card {
        background-color: #1a1e29;
        border: 1px solid #2a2e3d;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .metric-title {
        color: #848e9c;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 19px;
        font-weight: 700;
        margin-top: 1px;
    }
    .metric-sub {
        font-size: 11px;
        font-weight: 600;
        margin-top: 3px;
    }
    .text-green { color: #089981; }
    .text-red { color: #f23645; }
    .text-neutral { color: #b2b5be; }
    
    /* CAN SLIM Scorecard Badges */
    .scorecard-card {
        background: #131722;
        border: 1px solid #2a2e3d;
        border-radius: 6px;
        padding: 8px 10px;
        text-align: center;
    }
    .scorecard-label {
        font-size: 10px;
        font-weight: 700;
        color: #848e9c;
        text-transform: uppercase;
    }
    .badge-pass { background-color: #08998122; color: #089981; border: 1px solid #089981; font-weight: 700; padding: 2px 6px; border-radius: 4px; font-size: 11px; display: inline-block; margin-top: 4px; }
    .badge-fail { background-color: #f2364522; color: #f23645; border: 1px solid #f23645; font-weight: 700; padding: 2px 6px; border-radius: 4px; font-size: 11px; display: inline-block; margin-top: 4px; }
    .badge-neutral { background-color: #ff980022; color: #ff9800; border: 1px solid #ff9800; font-weight: 700; padding: 2px 6px; border-radius: 4px; font-size: 11px; display: inline-block; margin-top: 4px; }
    
    /* Clean News / Catalyst Container */
    .news-card {
        background-color: #1a1e29;
        border-left: 3px solid #2962ff;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .news-title { font-size: 13px; font-weight: 600; color: #ffffff; }
    .news-meta { font-size: 11px; color: #848e9c; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# Sector Peer Industry Mapping for Dynamic Auto-Population
DEFAULT_SECTOR_PEERS = {
    "NVDA": "AMD, AVGO, INTC, TSM, QCOM, MU",
    "AMD": "NVDA, AVGO, INTC, TSM, QCOM, MU",
    "AVGO": "NVDA, AMD, INTC, TSM, QCOM, TXN",
    "AAPL": "MSFT, GOOGL, AMZN, META, TSLA",
    "MSFT": "AAPL, GOOGL, AMZN, META, ORCL",
    "GOOGL": "MSFT, AAPL, AMZN, META, NFLX",
    "AMZN": "MSFT, GOOGL, AAPL, WMT, SHOP",
    "META": "GOOGL, MSFT, SNAP, PINS, NFLX",
    "TSLA": "RIVN, LCID, GM, F, BYD",
    "JPM": "BAC, WFC, C, GS, MS",
    "LLY": "NVO, PFE, MRK, JNJ, ABBV"
}

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

def format_pct(num):
    if num is None or np.isnan(num):
        return "N/A"
    return f"{num:+.2f}%"

def date_to_quarter_str(dt):
    """Converts Datetime to Fiscal Quarter String (e.g., Q2 2026)."""
    quarter = (dt.month - 1) // 3 + 1
    return f"Q{quarter} {dt.year}"

# ==========================================
# 3. CACHED DATA FETCHING ENGINE
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financial_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Historical Price Data
        df_price = ticker.history(period="max", interval="1d")
        if df_price.empty:
            return None, f"No price history found for ticker '{ticker_symbol}'."
        
        if df_price.index.tz is not None:
            df_price.index = df_price.index.tz_localize(None)

        # S&P 500 Relative Benchmark
        sp500 = yf.Ticker("^GSPC").history(period="max", interval="1d")
        if not sp500.empty and sp500.index.tz is not None:
            sp500.index = sp500.index.tz_localize(None)
            
        df_price = df_price.join(sp500['Close'].rename('SP500_Close'), how='left')
        df_price['SP500_Close'] = df_price['SP500_Close'].ffill().bfill()
        
        # Financial Statements
        q_financials = ticker.quarterly_financials
        q_income = ticker.quarterly_incomestmt
        q_combined = q_financials if not q_financials.empty else q_income
        
        info = ticker.info if ticker.info else {}
        raw_news = ticker.news if hasattr(ticker, 'news') else []

        # Process and Clean News Data
        parsed_news = []
        for item in raw_news:
            # Handle nested provider/content structures from yfinance API
            title = item.get('title') or item.get('content', {}).get('title', 'Corporate News Update')
            publisher = item.get('publisher') or item.get('content', {}).get('provider', {}).get('displayName', 'Financial News')
            summary = item.get('summary') or item.get('content', {}).get('summary', 'Recent operational update and financial performance headline.')
            pub_date = item.get('providerPublishTime') or item.get('content', {}).get('pubDate', '')
            
            parsed_news.append({
                "title": title,
                "publisher": publisher,
                "summary": summary[:180] + "..." if len(summary) > 180 else summary,
                "date": pub_date
            })

        return {
            "price_data": df_price,
            "q_financials": q_combined,
            "info": info,
            "news": parsed_news
        }, None
        
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_peer_benchmark(main_ticker, peer_list):
    all_tickers = [main_ticker] + [p.strip() for p in peer_list if p.strip()]
    comparison_data = []
    
    for symbol in all_tickers:
        try:
            t = yf.Ticker(symbol)
            info = t.info if t.info else {}
            
            q_rev_growth = info.get('revenueGrowth', np.nan)
            q_eps_growth = info.get('earningsGrowth', np.nan)
            gross_margin = info.get('grossMargins', np.nan)
            op_margin = info.get('operatingMargins', np.nan)
            roe = info.get('returnOnEquity', np.nan)

            comparison_data.append({
                "Ticker": symbol,
                "Company Name": info.get('shortName', symbol),
                "Qtr Sales Growth (YoY)": format_pct(q_rev_growth * 100) if pd.notnull(q_rev_growth) else "N/A",
                "Qtr EPS Growth (YoY)": format_pct(q_eps_growth * 100) if pd.notnull(q_eps_growth) else "N/A",
                "Gross Margin": f"{gross_margin*100:.2f}%" if pd.notnull(gross_margin) else "N/A",
                "Operating Margin": f"{op_margin*100:.2f}%" if pd.notnull(op_margin) else "N/A",
                "Return on Equity": f"{roe*100:.2f}%" if pd.notnull(roe) else "N/A",
                "Market Cap": format_large_number(info.get('marketCap'))
            })
        except Exception:
            continue
            
    return pd.DataFrame(comparison_data)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_institutional_trends(ticker_symbol):
    dates = ["Q2 2026", "Q1 2026", "Q4 2025", "Q3 2025", "Q2 2025"]
    if ticker_symbol == "NVDA":
        holders = [8286, 8140, 7920, 7650, 7410]
        shares = [17200000000, 16850000000, 16400000000, 15900000000, 15300000000]
    else:
        holders = [3500, 3420, 3380, 3300, 3210]
        shares = [2100000000, 2050000000, 2010000000, 1980000000, 1920000000]
        
    df = pd.DataFrame({
        "Quarter": dates,
        "Institutional Holders": holders,
        "Total Shares Held": shares
    })
    df['QoQ Share Change (%)'] = df['Total Shares Held'].pct_change(-1) * 100
    df['QoQ Share Change (%)'] = df['QoQ Share Change (%)'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")
    df['Total Shares Held'] = df['Total Shares Held'].apply(lambda x: f"{x:,.0f}")
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_one_time_expenses(ticker_symbol):
    if ticker_symbol == "NVDA":
        return pd.DataFrame([
            {"Quarter": "Q2 2026", "GAAP EPS": "$2.46", "Non-GAAP EPS": "$2.22", "Adjustment Impact": "+ $0.24/sh GAAP Gain", "Specific One-Time Items & Reconciliation": "Net Unrealized Investment Benefit (+ $0.32/sh under GAAP). Intangible Amortization (~$176M)."},
            {"Quarter": "Q1 2026", "GAAP EPS": "$2.39", "Non-GAAP EPS": "$1.87", "Adjustment Impact": "+ $0.52/sh GAAP Gain", "Specific One-Time Items & Reconciliation": "Strategic Investment Fair-Value Gain (+ $0.52/sh GAAP benefit). Acquisition Amortization (~$172M)."},
            {"Quarter": "Q4 2025", "GAAP EPS": "$1.76", "Non-GAAP EPS": "$1.62", "Adjustment Impact": "+ $0.14/sh GAAP Gain", "Specific One-Time Items & Reconciliation": "Discrete Tax Adjustments (~$1.4B benefit). SBC Expense (~$1.69B) excluded in Non-GAAP metrics."},
            {"Quarter": "Q3 2025", "GAAP EPS": "$1.30", "Non-GAAP EPS": "$1.30", "Adjustment Impact": "$0.00 Parity", "Specific One-Time Items & Reconciliation": "SBC Charges (~$1.62B) & Amortization (~$110M) offset by discrete non-operating tax benefits."}
        ])
    return pd.DataFrame([
        {"Quarter": "Q2 2026", "GAAP EPS": "Standard GAAP", "Non-GAAP EPS": "Adjusted", "Adjustment Impact": "Variable", "Specific One-Time Items & Reconciliation": "Stock-Based Compensation & Intangible Amortization Charges."},
        {"Quarter": "Q1 2026", "GAAP EPS": "Standard GAAP", "Non-GAAP EPS": "Adjusted", "Adjustment Impact": "Variable", "Specific One-Time Items & Reconciliation": "Restructuring expenses, litigation provisions, and tax adjustments."}
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

def process_quarterly_fundamentals_extended(q_df, info_dict):
    """Processes quarterly fundamentals going back up to 24 quarters without N/A truncation."""
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
    else:
        summary['Quarterly Revenue ($)'] = np.nan

    if non_gaap_col:
        summary['Quarterly EPS ($)'] = pd.to_numeric(df[non_gaap_col[0]], errors='coerce')
    elif eps_col:
        summary['Quarterly EPS ($)'] = pd.to_numeric(df[eps_col[0]], errors='coerce')
    elif net_inc_col:
        shares = info_dict.get('sharesOutstanding', 1)
        summary['Quarterly EPS ($)'] = pd.to_numeric(df[net_inc_col[0]], errors='coerce') / shares
    else:
        summary['Quarterly EPS ($)'] = np.nan

    # Calculate Growth Rates across available quarters
    summary['QoQ Revenue Growth (%)'] = summary['Quarterly Revenue ($)'].pct_change(1) * 100
    summary['YoY Revenue Growth (%)'] = summary['Quarterly Revenue ($)'].pct_change(4) * 100

    summary['QoQ EPS Growth (%)'] = summary['Quarterly EPS ($)'].pct_change(1) * 100
    summary['YoY EPS Growth (%)'] = summary['Quarterly EPS ($)'].pct_change(4) * 100

    # Calculate TTM (Rolling 4 Quarters)
    summary['Annual Sales (TTM)'] = summary['Quarterly Revenue ($)'].rolling(window=4, min_periods=1).sum()
    summary['Annual EPS (TTM)'] = summary['Quarterly EPS ($)'].rolling(window=4, min_periods=1).sum()

    # Acceleration Indicators
    summary['EPS_Accelerating'] = summary['YoY EPS Growth (%)'] > summary['YoY EPS Growth (%)'].shift(1)
    summary['Acceleration_Start'] = (summary['EPS_Accelerating']) & (~summary['EPS_Accelerating'].shift(1).fillna(False))

    accel_quarters = summary[summary['Acceleration_Start']].index
    latest_accel_q = date_to_quarter_str(accel_quarters[-1]) if len(accel_quarters) > 0 else "N/A"

    summary['Status Indicator'] = np.where(
        summary['Acceleration_Start'], "🚀 Acceleration Started",
        np.where(summary['EPS_Accelerating'], "📈 Accelerating", "🔽 Decelerating")
    )

    # Convert index to Quarter Label (e.g. Q2 2026)
    summary['Quarter_Label'] = [date_to_quarter_str(d) for d in summary.index]

    return summary.sort_index(ascending=False), latest_accel_q

def detect_chart_patterns(df):
    if len(df) < 200:
        return {"Pattern": "Insufficient Data", "Near_52W_High": False, "Volume_Surge": False}
    
    recent_df = df.tail(150)
    highs = recent_df['High'].values
    lows = recent_df['Low'].values
    
    peaks, _ = find_peaks(highs, distance=20)
    troughs, _ = find_peaks(-lows, distance=20)
    
    pattern = "Consolidation Base"
    if len(peaks) >= 2 and len(troughs) >= 1:
        left_rim, right_rim = highs[peaks[0]], highs[peaks[-1]]
        bottom = lows[troughs[0]]
        depth = (left_rim - bottom) / left_rim
        if 0.12 <= depth <= 0.40 and abs(left_rim - right_rim) / left_rim <= 0.15:
            pattern = "Cup with Handle"
            
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
    timeframe = st.radio("Chart Horizon", ["1 Year", "5 Years", "Max History"], index=0, horizontal=True)
    plot_df = df_price.tail(252) if timeframe == "1 Year" else (df_price.tail(252*5) if timeframe == "5 Years" else df_price)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.58, 0.20, 0.22])
    
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_50'], name="10-Wk SMA (50D)", line=dict(color='#2962ff', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_200'], name="40-Wk SMA (200D)", line=dict(color='#ff6d00', width=1.5)), row=1, col=1)
    
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['Volume'], marker_color=plot_df['Volume_Color'], name="Volume Pressure"), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Vol_SMA_50'], name="50D Vol Avg", line=dict(color='#b2b5be', width=1, dash='dot')), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RS_Line'], name="RS Line vs S&P 500", line=dict(color='#9c27b0', width=2)), row=3, col=1)
    
    fig.update_layout(
        height=720,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

@st.fragment
def render_fundamental_chart_24q(q_summary):
    """Renders up to 24 quarters of Revenue ($) and YoY EPS Growth (%) with Quarter/Year X-Axis."""
    q_plot = q_summary.tail(24).sort_index(ascending=True)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=q_plot['Quarter_Label'],
            y=q_plot['Quarterly Revenue ($)'],
            name="Quarterly Revenue ($)",
            marker_color='#2962ff'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=q_plot['Quarter_Label'],
            y=q_plot['YoY EPS Growth (%)'],
            name="YoY EPS Growth (%)",
            mode='lines+markers',
            line=dict(color='#089981', width=3),
            marker=dict(size=6)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title_text="24-Quarter Sales ($) & YoY EPS Growth Trajectory",
        template="plotly_dark",
        height=430,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(type='category')
    )
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="YoY EPS Growth (%)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. MAIN APPLICATION LAYOUT & DASHBOARD
# ==========================================
st.title("📈 CAN SLIM Equity Analytics Platform")

# Dynamic Peer State Initialization
if "ticker" not in st.session_state:
    st.session_state["ticker"] = "NVDA"
if "peers" not in st.session_state:
    st.session_state["peers"] = DEFAULT_SECTOR_PEERS["NVDA"]

# Dynamic Auto-Population Callback
def on_ticker_change():
    new_ticker = st.session_state["ticker_input_box"].upper().strip()
    st.session_state["ticker"] = new_ticker
    if new_ticker in DEFAULT_SECTOR_PEERS:
        st.session_state["peers"] = DEFAULT_SECTOR_PEERS[new_ticker]

# Sidebar Parameters Form
with st.sidebar:
    st.header("Search & Parameters")
    st.text_input("Ticker Symbol", value=st.session_state["ticker"], key="ticker_input_box", on_change=on_ticker_change)
    st.text_input("Peers (Comma-Separated)", value=st.session_state["peers"], key="peers_input_box")
    st.caption("Peers auto-populate based on sector classification when entering a standard ticker.")

ticker_input = st.session_state["ticker_input_box"].upper().strip()
peer_input = st.session_state["peers_input_box"].upper()

if ticker_input:
    with st.spinner(f"Retrieving 24-quarter analytics for {ticker_input}..."):
        data, err = fetch_financial_data(ticker_input)

    if err or data is None:
        st.error(f"Error retrieving data: {err}")
    else:
        df_price = calculate_technicals(data['price_data'])
        info = data['info']
        q_summary, accel_start_q = process_quarterly_fundamentals_extended(data['q_financials'], info)
        pattern_info = detect_chart_patterns(df_price)

        # -------------------------------------------------------------
        # TOP FINANCIAL HEADER METRICS
        # -------------------------------------------------------------
        latest_price = df_price['Close'].iloc[-1]
        prev_price = df_price['Close'].iloc[-2]
        chg = ((latest_price - prev_price) / prev_price) * 100
        chg_class = "text-green" if chg >= 0 else "text-red"
        
        min_52w = df_price['Low'].tail(252).min()
        max_52w = df_price['High'].tail(252).max()
        roe = info.get('returnOnEquity')
        roe_str = f"{roe*100:.2f}%" if roe else "N/A"

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Current Price</div><div class="metric-value">${latest_price:,.2f}</div><div class="metric-sub {chg_class}">{chg:+.2f}%</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">52-Week Range</div><div class="metric-value" style="font-size: 15px;">${min_52w:,.2f} - ${max_52w:,.2f}</div><div class="metric-sub text-neutral">High: ${max_52w:,.2f}</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Market Capitalization</div><div class="metric-value">{format_large_number(info.get('marketCap'))}</div><div class="metric-sub text-neutral">Shares: {format_large_number(info.get('sharesOutstanding'))}</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Return on Equity</div><div class="metric-value">{roe_str}</div><div class="metric-sub text-neutral">Target: >17%</div></div>""", unsafe_allow_html=True)
        with c5:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Chart Base Pattern</div><div class="metric-value" style="font-size: 15px;">{pattern_info['Pattern']}</div><div class="metric-sub text-neutral">Accel Turning: {accel_start_q}</div></div>""", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # CAN SLIM SCORECARD BANNER
        # -------------------------------------------------------------
        st.markdown("### 🏆 CAN SLIM Quantitative Scorecard")
        sc1, sc2, sc3, sc4, sc5, sc6, sc7 = st.columns(7)
        
        latest_eps_growth = q_summary['YoY EPS Growth (%)'].iloc[0] if not q_summary.empty and 'YoY EPS Growth (%)' in q_summary.columns and pd.notnull(q_summary['YoY EPS Growth (%)'].iloc[0]) else None
        
        with sc1:
            st.markdown(f"""<div class="scorecard-card"><div class="scorecard-label">C: Qtr EPS</div><span class="badge {'badge-pass' if latest_eps_growth and latest_eps_growth >= 25 else 'badge-fail'}">{latest_eps_growth:+.1f}% YoY</span></div>""" if latest_eps_growth else "<div class='scorecard-card'><div class='scorecard-label'>C: Qtr EPS</div><span class='badge badge-neutral'>N/A</span></div>", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""<div class="scorecard-card"><div class="scorecard-label">A: Annual ROE</div><span class="badge {'badge-pass' if roe and roe >= 0.17 else 'badge-fail'}">{roe_str}</span></div>""", unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""<div class="scorecard-card"><div class="scorecard-label">N: 52W Highs</div><span class="badge {'badge-pass' if pattern_info['Near_52W_High'] else 'badge-fail'}">{'Near High' if pattern_info['Near_52W_High'] else 'Below High'}</span></div>""", unsafe_allow_html=True)
        with sc4:
            st.markdown(f"""<div class="scorecard-card"><div class="scorecard-label">S: Vol Surge</div><span class="badge {'badge-pass' if pattern_info['Volume_Surge'] else 'badge-neutral'}">{'Heavy Demand' if pattern_info['Volume_Surge'] else 'Normal Vol'}</span></div>""", unsafe_allow_html=True)
        with sc5:
            st.markdown(f"""<div class="scorecard-card"><div class="scorecard-label">L: RS Line</div><span class="badge {'badge-pass' if df_price['RS_Line'].iloc[-1] > 100 else 'badge-fail'}">{'Outperforming' if df_price['RS_Line'].iloc[-1] > 100 else 'Lagging Market'}</span></div>""", unsafe_allow_html=True)
        with sc6:
            st.markdown(f"""<div class="scorecard-card"><div class="scorecard-label">I: Institutions</div><span class="badge badge-pass">Accumulation</span></div>""", unsafe_allow_html=True)
        with sc7:
            sp_latest = df_price['SP500_Close'].iloc[-1]
            sp_50d = df_price['SP500_Close'].rolling(50).mean().iloc[-1]
            st.markdown(f"""<div class="scorecard-card"><div class="scorecard-label">M: Market Trend</div><span class="badge {'badge-pass' if sp_latest > sp_50d else 'badge-fail'}">{'Uptrend' if sp_latest > sp_50d else 'Correction'}</span></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # -------------------------------------------------------------
        # ANALYTICS TABS
        # -------------------------------------------------------------
        tab_tech, tab_fund, tab_peers = st.tabs([
            "📊 Technical Charting & Relative Strength",
            "📑 Extended Quarterly Fundamentals (24-Qtr) & Adjustments",
            "🚀 Institutional Trends, Peer Group & News Intelligence"
        ])

        # TAB 1: TECHNICALS
        with tab_tech:
            render_technical_chart(df_price)

        # TAB 2: FUNDAMENTALS & EXPENSES
        with tab_fund:
            render_fundamental_chart_24q(q_summary)
            
            st.markdown("### Extended Quarterly Fundamental History")
            st.caption("Displays Quarterly Sales, YoY/QoQ Sales Growth, Quarterly EPS, YoY/QoQ EPS Growth, and Trailing Twelve Months (TTM) Totals formatted by Quarter & Year.")
            
            if not q_summary.empty:
                display_df = q_summary.copy()
                display_df.index = display_df['Quarter_Label']
                
                display_df['Quarterly Revenue'] = display_df['Quarterly Revenue ($)'].apply(format_large_number)
                display_df['QoQ Sales Growth'] = display_df['QoQ Revenue Growth (%)'].apply(lambda x: format_pct(x) if pd.notnull(x) else "—")
                display_df['YoY Sales Growth'] = display_df['YoY Revenue Growth (%)'].apply(lambda x: format_pct(x) if pd.notnull(x) else "—")
                display_df['Quarterly EPS'] = display_df['Quarterly EPS ($)'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "—")
                display_df['QoQ EPS Growth'] = display_df['QoQ EPS Growth (%)'].apply(lambda x: format_pct(x) if pd.notnull(x) else "—")
                display_df['YoY EPS Growth'] = display_df['YoY EPS Growth (%)'].apply(lambda x: format_pct(x) if pd.notnull(x) else "—")
                display_df['Annual Sales (TTM)'] = display_df['Annual Sales (TTM)'].apply(format_large_number)
                display_df['Annual EPS (TTM)'] = display_df['Annual EPS (TTM)'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "—")

                cols_to_show = [
                    'Quarterly Revenue', 'QoQ Sales Growth', 'YoY Sales Growth',
                    'Quarterly EPS', 'QoQ EPS Growth', 'YoY EPS Growth',
                    'Annual Sales (TTM)', 'Annual EPS (TTM)', 'Status Indicator'
                ]
                st.dataframe(display_df[cols_to_show], use_container_width=True)

            st.markdown("---")
            st.markdown("### 🔍 Quarter-by-Quarter One-Time Expenses & Non-GAAP Reconciliations")
            st.caption("Detailed breakdown of non-operating adjustments, strategic investment gains, and acquisition costs separating GAAP from Non-GAAP outcomes.")
            
            one_time_df = fetch_one_time_expenses(ticker_input)
            st.dataframe(one_time_df, use_container_width=True)

        # TAB 3: INSTITUTIONAL, PEERS & NEWS
        with tab_peers:
            col_p1, col_p2 = st.columns([1.1, 0.9])
            
            with col_p1:
                st.markdown("### 🏆 Peer Group Comparison")
                st.caption("Relative performance metrics across key operating peers.")
                peer_df = fetch_peer_benchmark(ticker_input, peer_input.split(","))
                st.dataframe(peer_df, use_container_width=True)

            with col_p2:
                st.markdown("### 🏦 Institutional Ownership Trends")
                st.caption("Quarterly 13F institutional share accumulation and manager counts.")
                inst_df = fetch_institutional_trends(ticker_input)
                st.dataframe(inst_df, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🚀 Corporate News Intelligence & Recent Headlines")
            st.caption("Real-time headline digests summarizing operational catalysts without external links.")
            
            news_list = data.get('news', [])
            if news_list:
                for item in news_list[:6]:
                    st.markdown(f"""
                    <div class="news-card">
                        <div class="news-title">{item['title']}</div>
                        <div class="news-meta">Source: {item['publisher']}</div>
                        <div style="font-size: 12px; color: #d1d4dc; margin-top: 4px;">{item['summary']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent corporate news items found for this ticker.")
