import streamlit as st
import json
from pathlib import Path
def display_live_metrics():
    cache_path = Path("cache/live_data.json")
    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        st.subheader("📊 Live Trading Metrics (Simulation)")
        st.metric("Prix LTC/USDC", f"{data['price']} $")
        st.metric("Solde USDC", f"{data['usdc_balance']} $")
        st.metric("Solde LTC", f"{data['ltc_balance']} LTC")
        st.metric("PnL", f"{data['pnl']} $")
        st.metric("Drawdown", f"{data['drawdown']} %")
    else:
        st.warning("Aucune donnée de simulation disponible.")
