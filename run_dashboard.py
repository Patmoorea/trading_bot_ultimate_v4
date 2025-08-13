import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
# Import des composants
from dashboard.main_dashboard import EnhancedTradingDashboard, NotificationManager
from dashboard.strategies.quantum_strat import QuantumStrategy
# Initialisation
dashboard = EnhancedTradingDashboard()
notifications = NotificationManager()
quantum_strat = QuantumStrategy()
# Configuration de la page
st.set_page_config(
    page_title="Trading Bot Ultimate v3",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    # Mode de trading
    mode = st.radio(
        "Mode de Trading",
        ["Live Trading", "Paper Trading", "Backtest"],
        index=1
    )
    # Sélection des paires
    st.subheader("Paires de Trading")
    pairs = {
        "BTC/USD": st.checkbox("BTC/USD", value=True),
        "ETH/USD": st.checkbox("ETH/USD", value=True),
        "SOL/USD": st.checkbox("SOL/USD")
    }
    # Paramètres de risque
    st.subheader("Gestion des Risques")
    leverage = st.slider("Levier", 1, 10, 1)
    position_size = st.slider("Taille Position (%)", 1, 100, 10)
    # Stratégies actives
    st.subheader("Stratégies")
    strategies = {
        "Quantum Forest": st.checkbox("Quantum Forest", value=True),
        "Anti-Frontrun": st.checkbox("Anti-Frontrunning", value=True),
        "Multi-Timeframe": st.checkbox("Multi-Timeframe", value=False)
    }
# Corps principal
st.title("📈 Trading Bot Ultimate v3")
# Métriques principales
col1, col2, col3, col4 = st.columns(4)
# Simulation de données pour la démo
metrics = {
    'daily_pnl': 3.5,
    'total_pnl': 15.5,
    'win_rate': 65.0,
    'drawdown': -2.5,
    'sharpe_ratio': 1.8,
    'trades_count': 42,
    'active_positions': 3
}
dashboard.update_metrics(metrics)
with col1:
    st.metric(
        "PnL Journalier",
        f"{dashboard.metrics.daily_pnl}%",
        f"{dashboard.metrics.daily_pnl}%"
    )
with col2:
    st.metric(
        "Win Rate",
        f"{dashboard.metrics.win_rate}%",
        "1.2%"
    )
with col3:
    st.metric(
        "Drawdown",
        f"{dashboard.metrics.drawdown}%",
        "-0.5%"
    )
with col4:
    st.metric(
        "Positions Actives",
        dashboard.metrics.active_positions,
        "2"
    )
# Tabs pour l'organisation
tab1, tab2, tab3, tab4 = st.tabs([
    "Trading View",
    "Positions & Ordres",
    "Performance",
    "Système"
])
with tab1:
    # Graphique principal
    chart_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 101
    fig = go.Figure(data=[go.Candlestick(
        x=chart_data.index,
        open=chart_data['open'],
        high=chart_data['high'],
        low=chart_data['low'],
        close=chart_data['close']
    )])
    fig.update_layout(
        title="BTC/USD",
        yaxis_title="Prix",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    # Signaux de la stratégie quantum
    signal, confidence = quantum_strat.get_signal()
    st.info(f"Signal Quantum : {signal} (Confiance: {confidence*100:.1f}%)")
with tab2:
    # Positions actives
    st.subheader("Positions Actives")
    positions_df = pd.DataFrame({
        'Paire': ['BTC/USD', 'ETH/USD'],
        'Type': ['Long', 'Short'],
        'Entrée': [35000, 2200],
        'Taille': [0.5, 5],
        'PnL': ['2.5%', '-1.2%']
    })
    st.dataframe(positions_df, use_container_width=True)
    # Ordres en attente
    st.subheader("Ordres en Attente")
    orders_df = pd.DataFrame({
        'Paire': ['BTC/USD'],
        'Type': ['Limit Buy'],
        'Prix': [34500],
        'Taille': [0.3]
    })
    st.dataframe(orders_df, use_container_width=True)
with tab3:
    # Performance
    st.subheader("Analyse de Performance")
    # Courbe de capital
    perf_data = pd.DataFrame({
        'Capital': np.random.randn(100).cumsum() + 100
    st.line_chart(perf_data)
    # Métriques de performance
    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("Sharpe Ratio", "1.8", "0.2")
    with col6:
        st.metric("Sortino Ratio", "2.1", "0.3")
    with col7:
        st.metric("Calmar Ratio", "3.2", "0.1")
with tab4:
    # Statut système
    st.subheader("État du Système")
    col8, col9 = st.columns(2)
    with col8:
        st.write("🟢 API Exchange: Connecté")
        st.write("🟢 Base de données: Synchronisée")
        st.write("🟢 GPU: Actif")
    with col9:
        st.write("Mémoire utilisée: 45%")
        st.progress(45)
        st.write("CPU: 30%")
        st.progress(30)
# Notifications
notifications.add_alert("Signal d'achat détecté sur BTC/USD", "info")
notifications.add_alert("Risque élevé détecté", "warning")
active_alerts = notifications.get_active_alerts()
if active_alerts:
    for alert in active_alerts:
        if alert['level'] == "warning":
            st.sidebar.warning(alert['message'])
        else:
            st.sidebar.info(alert['message'])
# Rafraîchissement automatique
st.empty()
