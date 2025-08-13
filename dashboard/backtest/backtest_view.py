import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
class BacktestView:
    def __init__(self):
        self.fig = go.Figure()
    def render_controls(self) -> Dict:
        """Affiche les contrôles du backtest"""
        st.subheader("Configuration Backtest")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Date de début",
            )
            timeframe = st.selectbox(
                "Timeframe",
                ["1m", "5m", "15m", "1h", "4h", "1d"]
            )
        with col2:
            end_date = st.date_input(
                "Date de fin",
            )
            capital = st.number_input(
                "Capital Initial",
                min_value=100,
                value=10000
            )
        # Paramètres avancés dans un expander
        with st.expander("Paramètres Avancés"):
            col3, col4 = st.columns(2)
            with col3:
                commission = st.slider(
                    "Commission (%)",
                    0.0, 1.0, 0.1
                )
                slippage = st.slider(
                    "Slippage (%)",
                    0.0, 1.0, 0.1
                )
            with col4:
                leverage = st.slider(
                    "Levier",
                    1, 10, 1
                )
                risk_per_trade = st.slider(
                    "Risque par trade (%)",
                    1, 10, 2
                )
        return {
            'start_date': start_date,
            'end_date': end_date,
            'timeframe': timeframe,
            'capital': capital,
            'commission': commission,
            'slippage': slippage,
            'leverage': leverage,
            'risk_per_trade': risk_per_trade
        }
    def render_results(self, results: Dict):
        """Affiche les résultats du backtest"""
        st.subheader("Résultats du Backtest")
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Rendement Total",
                value=f"{results.get('total_return', 0):.2f}%",
                delta=f"{results.get('return_change', 0):.2f}%"
            )
        with col2:
            st.metric(
                label="Sharpe Ratio",
                value=f"{results.get('sharpe', 0):.2f}",
                delta=None
            )
        with col3:
            st.metric(
                label="Max Drawdown",
                value=f"{results.get('max_drawdown', 0):.2f}%",
                delta=None
            )
        with col4:
            st.metric(
                label="Win Rate",
                value=f"{results.get('win_rate', 0):.1f}%",
                delta=None
            )
        # Graphique de performance
        equity_curve = results.get('equity_curve', pd.Series())
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                name="Equity"
            )
        )
        fig.update_layout(
            title="Courbe de Capital",
            xaxis_title="Date",
            yaxis_title="Capital",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        # Statistiques détaillées dans un expander
        with st.expander("Statistiques Détaillées"):
            stats = results.get('stats', {})
            col5, col6 = st.columns(2)
            with col5:
                st.write("Métriques de Performance")
                st.write(f"Alpha: {stats.get('alpha', 0):.4f}")
                st.write(f"Beta: {stats.get('beta', 0):.4f}")
                st.write(f"Sortino: {stats.get('sortino', 0):.4f}")
                st.write(f"Calmar: {stats.get('calmar', 0):.4f}")
            with col6:
                st.write("Métriques de Trading")
                st.write(f"Nombre de trades: {stats.get('n_trades', 0)}")
                st.write(f"Profit moyen: {stats.get('avg_profit', 0):.2f}%")
                st.write(f"Perte moyenne: {stats.get('avg_loss', 0):.2f}%")
                st.write(f"Taille moyenne position: {stats.get('avg_size', 0):.2f}")
