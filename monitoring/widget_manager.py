import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any
import pandas as pd
import numpy as np
class EnhancedWidgetManager:
    def __init__(self):
        self.widgets: Dict[str, Any] = {}
        self.layout = self._create_layout()
    def _create_layout(self):
        st.set_page_config(layout="wide")
        # Création du layout en colonnes
        col1, col2, col3 = st.columns(3)
        with col1:
            self.widgets['market_overview'] = st.empty()
            self.widgets['portfolio_status'] = st.empty()
            self.widgets['active_orders'] = st.empty()
        with col2:
            self.widgets['performance_metrics'] = st.empty()
            self.widgets['risk_indicators'] = st.empty()
            self.widgets['news_feed'] = st.empty()
        with col3:
            self.widgets['alerts'] = st.empty()
            self.widgets['system_status'] = st.empty()
            self.widgets['voice_commands'] = st.empty()
        return {
            'col1': col1,
            'col2': col2,
            'col3': col3
        }
    def update_market_overview(self, data: Dict):
        fig = go.Figure()
        # Ajout des données de prix
        fig.add_trace(go.Candlestick(
            x=data['timestamps'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close']
        ))
        # Ajout des volumes
        fig.add_trace(go.Bar(
            x=data['timestamps'],
            y=data['volume'],
            name="Volume",
            yaxis="y2"
        ))
        # Mise en page
        fig.update_layout(
            title="Market Overview",
            yaxis_title="Price",
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right"
            )
        )
        self.widgets['market_overview'].plotly_chart(fig, use_container_width=True)
    def update_portfolio(self, portfolio: Dict):
        df = pd.DataFrame(portfolio['holdings'])
        fig = go.Figure(data=[
            go.Pie(labels=df['asset'],
                  values=df['value_usd'],
                  textinfo='label+percent',
                  hole=.3)
        ])
        self.widgets['portfolio_status'].plotly_chart(fig, use_container_width=True)
    def update_risk_indicators(self, risk_data: Dict):
        fig = go.Figure()
        # Ajout des indicateurs de risque
        for indicator, value in risk_data.items():
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': indicator},
                gauge={'axis': {'range': [0, 100]}}
            ))
        fig.update_layout(
            grid={'rows': 2, 'columns': 2, 'pattern': "independent"}
        )
        self.widgets['risk_indicators'].plotly_chart(fig, use_container_width=True)
    def update_news_feed(self, news: List[Dict]):
        df = pd.DataFrame(news)
        self.widgets['news_feed'].dataframe(df)
    def update_system_status(self, status: Dict):
        color_map = {
            'healthy': 'green',
            'warning': 'orange',
            'error': 'red'
        }
        status_md = ""
        for component, details in status.items():
            status_md += f"### {component}\n"
            status_md += f"Status: :{color_map[details['status']]}[{details['status']}]\n"
            status_md += f"Details: {details['message']}\n\n"
        self.widgets['system_status'].markdown(status_md)
    def update_voice_commands(self, commands: List[str]):
        commands_md = "### Available Voice Commands\n"
        for cmd in commands:
            commands_md += f"- {cmd}\n"
        self.widgets['voice_commands'].markdown(commands_md)
