import streamlit as st
import redis
import time
import plotly.graph_objs as go
import os
from dotenv import load_dotenv
load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
SYMBOL = "ltcusdt"
st.set_page_config(page_title="Dashboard Trading Ultimate", layout="wide")
st.title("🧠 Tableau de bord du bot de trading en temps réel")
price_placeholder = st.empty()
chart_placeholder = st.empty()
prices = []
for i in range(100):
    price = r.get(f"{SYMBOL}_last_price")
    if price:
        price = float(price)
        prices.append(price)
        price_placeholder.metric(f"Prix {SYMBOL.upper()}", f"{price:.2f} USDT")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=prices, mode='lines+markers', name='Prix'))
        fig.update_layout(title=f"Prix {SYMBOL.upper()} en temps réel", yaxis_title='Prix (USDT)')
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    time.sleep(1)
