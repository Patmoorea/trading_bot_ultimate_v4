import plotly.graph_objs as go
import numpy as np

def plot_orderflow_heatmap(orderbook_matrix, title="Heatmap Orderflow"):
    fig = go.Figure(
        data=go.Heatmap(z=orderbook_matrix, colorscale="Jet")
    )
    fig.update_layout(
        title=title,
        xaxis_title="Prix",
        yaxis_title="Profondeur",
        template="plotly_dark"
    )
    return fig