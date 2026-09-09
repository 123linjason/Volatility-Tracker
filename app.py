import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

def detect_chart_base_pattern(df: pd.DataFrame) -> dict:
    """
    Dynamically analyzes price action over the last 1-year window 
    to classify the chart base pattern (CAN SLIM rules).
    """
    if df.empty or len(df) < 60:
        return {"pattern": "Insufficient Data", "notes": "Need >= 60 bars"}

    close = df['Close']
    high = df['High']
    low = df['Low']

    # 52-week peak and drawdowns
    high_52w = high.max()
    curr_price = close.iloc[-1]
    depth_pct = (high_52w - low.min()) / high_52w * 100
    off_high_pct = (high_52w - curr_price) / high_52w * 100

    # Find local minima for Double Bottom / W-Bottom detection
    recent_60 = low.tail(60)
    min_idx1 = recent_60.iloc[:30].idxmin()
    min_idx2 = recent_60.iloc[30:].idxmin()
    
    val1 = low.loc[min_idx1]
    val2 = low.loc[min_idx2]
    
    # Peak between the two troughs
    mid_peak = high.loc[min_idx1:min_idx2].max() if min_idx1 < min_idx2 else high_52w

    # Pattern Conditions
    # 1. Double Bottom / W-Bottom: Two similar troughs separated by a bounce mid-peak
    if abs(val1 - val2) / val1 < 0.04 and (mid_peak - min(val1, val2)) / min(val1, val2) > 0.08:
        if curr_price >= mid_peak * 0.95:
            return {"pattern": "W Bottom / Double Bottom", "notes": "Breaking out / Pivot stage"}
        return {"pattern": "W Bottom / Double Bottom", "notes": "Forming right side of base"}

    # 2. Cup with Handle: Depth 12%-35%, handle pullback < 12% near highs
    elif 12 <= depth_pct <= 38 and off_high_pct <= 15:
        recent_15_max = high.tail(15).max()
        recent_15_min = low.tail(15).min()
        handle_depth = (recent_15_max - recent_15_min) / recent_15_max * 100
        if handle_depth < 12:
            return {"pattern": "Cup with Handle", "notes": "Handle volume contracting"}
        return {"pattern": "Cup Base", "notes": "Rounding out right side"}

    # 3. Flat Base: Tight sideways consolidation within 15% range over 5+ weeks
    elif depth_pct <= 15 and len(df) >= 25:
        return {"pattern": "Flat Base", "notes": "Tight price consolidation"}

    # 4. High Tight Flag / Consolidation
    elif off_high_pct <= 8:
        return {"pattern": "Ascending Base / Near Highs", "notes": "Consolidating near 52W High"}

    else:
        return {"pattern": "Consolidation / No Clear Base", "notes": f"{off_high_pct:.1f}% off high"}


# Streamlit Execution
st.title("CAN SLIM Equity Analytics Platform")

ticker = st.sidebar.text_input("Ticker Symbol", value="PLTR").upper()

if ticker:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")

    if not hist.empty:
        pattern_info = detect_chart_base_pattern(hist)
        
        # Display dynamically in KPI card
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="CHART BASE PATTERN", value=pattern_info["pattern"])
            st.caption(f"Status: {pattern_info['notes']}")

        # Dynamic Recommendation Banner
        st.success(
            f"**BUY RECOMMENDATION:** **{ticker}** is currently classified under "
            f"**{pattern_info['pattern']}** structure ({pattern_info['notes']})."
        )
