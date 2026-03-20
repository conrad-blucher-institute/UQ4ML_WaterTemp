"""
V2 Layers Impact: box plot of metric grouped by num_layers, colored by activation.
Flexible sizing with legend.
"""

from typing import Optional
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_layers_impact_v2(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[go.Figure]:
    """
    Box plot: metric distribution by num_layers, colored by activation.
    Flexible width, proper legend.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    clean = data.dropna(subset=['leadtime', 'num_layers']).copy()
    clean['num_layers_str'] = clean['num_layers'].astype(int).astype(str) + ' layers'

    fig = px.box(
        clean,
        x='num_layers_str',
        y=metric_column,
        color='activation',
        title=f"Impact of Number of Layers on {metric_column}",
        labels={
            'num_layers_str': 'Number of Layers',
            metric_column: metric_column,
            'activation': 'Activation Function',
        },
        category_orders={
            'num_layers_str': sorted(
                clean['num_layers_str'].unique(),
                key=lambda x: int(x.split()[0]),
            ),
        },
    )

    fig.update_layout(
        height=550,
        legend=dict(title="Activation Function"),
        hovermode='closest',
    )

    if output_dir:
        fig.write_html(f"{output_dir}/05_layers_impact_v2.html")
        print("Saved: 05_layers_impact_v2.html")

    return fig
