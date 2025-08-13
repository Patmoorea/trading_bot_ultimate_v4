import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List
class QuantumView:
    def __init__(self):
        self.fig = go.Figure()
    def render_quantum_state(self, state_vector: List[complex]):
        """Affiche l'état quantique"""
        amplitudes = np.abs(state_vector) ** 2
        phases = np.angle(state_vector)
        # Bloch sphere visualization
        self.fig = go.Figure()
        self.fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[1],
            mode='markers',
            marker=dict(
                size=10,
                color='blue'
            ),
            name='Quantum State'
        ))
        self.fig.update_layout(
            title="État Quantique (Sphère de Bloch)",
            scene = dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        return self.fig
    def render_entanglement_metrics(self, metrics: Dict):
        """Affiche les métriques d'intrication"""
        st.subheader("Métriques d'Intrication")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Concurrence",
                value=f"{metrics.get('concurrence', 0):.3f}"
            )
        with col2:
            st.metric(
                label="Pureté",
                value=f"{metrics.get('purity', 0):.3f}"
            )
