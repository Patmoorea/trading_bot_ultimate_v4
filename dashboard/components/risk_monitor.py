import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List
import numpy as np
class RiskMonitor:
    def render_risk_metrics(self, metrics: Dict):
        """Affiche les métriques de risque"""
        st.subheader("Monitoring des Risques")
        # Value at Risk (VaR)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="VaR (95%)",
                value=f"{metrics.get('var_95', 0):.2f}%",
                delta=f"{metrics.get('var_change', 0):.2f}%",
                delta_color="inverse"
            )
        with col2:
            st.metric(
                label="CVaR",
                value=f"{metrics.get('cvar', 0):.2f}%",
                delta=f"{metrics.get('cvar_change', 0):.2f}%",
                delta_color="inverse"
            )
        with col3:
            st.metric(
                label="Beta",
                value=f"{metrics.get('beta', 0):.2f}",
                delta=None
            )
    def render_correlation_matrix(self, matrix: np.ndarray, assets: List[str]):
        """Affiche la matrice de corrélation"""
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=assets,
            y=assets,
            colorscale='RdBu'
        ))
        fig.update_layout(
            title="Matrice de Corrélation",
            xaxis_title="Assets",
            yaxis_title="Assets"
        )
        st.plotly_chart(fig, use_container_width=True)
