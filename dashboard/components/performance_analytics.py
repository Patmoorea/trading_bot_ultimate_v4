import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List
import numpy as np
class PerformanceAnalytics:
    def render_metrics(self, metrics: Dict):
        """Affiche les métriques de performance avancées"""
        st.subheader("Métriques de Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Sharpe Ratio",
                value=f"{metrics.get('sharpe', 0):.2f}",
                delta=f"{metrics.get('sharpe_change', 0):.2f}%"
            )
        with col2:
            st.metric(
                label="Sortino Ratio",
                value=f"{metrics.get('sortino', 0):.2f}",
                delta=f"{metrics.get('sortino_change', 0):.2f}%"
            )
        with col3:
            st.metric(
                label="Max Drawdown",
                value=f"{metrics.get('max_drawdown', 0):.2f}%",
                delta=f"{metrics.get('drawdown_change', 0):.2f}%",
                delta_color="inverse"
            )
        with col4:
            st.metric(
                label="Calmar Ratio",
                value=f"{metrics.get('calmar', 0):.2f}",
                delta=None
            )
    def render_underwater_plot(self, drawdowns: pd.Series):
        """Affiche le graphique underwater"""
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=drawdowns.index,
                y=drawdowns.values,
                fill='tozeroy',
                name="Drawdown"
            )
        )
        fig.update_layout(
            title="Analyse des Drawdowns",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig, use_container_width=True)
