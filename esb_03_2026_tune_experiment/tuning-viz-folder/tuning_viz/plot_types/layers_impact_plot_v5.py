"""
V5 Layers Impact: num_layers x-axis, colored by activation.
Single file with cycle dropdown.
"""

from typing import Optional
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .html_utils_v5 import wrap_figures_with_cycle_dropdown


def _build_layers_figure(data: pd.DataFrame, metric_column: str,
                          title_suffix: str = "") -> go.Figure:
    """Build a single layers impact figure."""
    clean = data.copy()
    clean['num_layers_cat'] = clean['num_layers'].astype(int).astype(str)
    layer_order = sorted(clean['num_layers_cat'].unique(), key=lambda x: int(x))

    fig = px.box(
        clean,
        x='num_layers_cat',
        y=metric_column,
        color='activation',
        title=f"Impact of Number of Layers on {metric_column}{title_suffix}",
        labels={
            'num_layers_cat': 'Number of Layers',
            metric_column: metric_column,
            'activation': 'Activation Function',
        },
        category_orders={'num_layers_cat': layer_order},
    )

    fig.update_layout(
        height=550,
        xaxis=dict(type='category'),
        legend=dict(title="Activation Function"),
        hovermode='closest',
    )

    return fig


def plot_layers_impact_v5(data: pd.DataFrame, metric_column: str,
                           output_dir: Optional[str] = None) -> Optional[str]:
    """
    Single HTML with cycle dropdown.
    Each view: num_layers on x, metric on y, colored by activation.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime', 'num_layers']).copy()
    cycles = sorted(clean['cycle'].unique())

    figure_htmls = {}

    # All cycles
    fig = _build_layers_figure(clean, metric_column, " (All Cycles)")
    figure_htmls["All Cycles"] = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Per cycle
    for cycle in cycles:
        c_int = int(cycle)
        cycle_data = clean[clean['cycle'] == cycle]
        fig = _build_layers_figure(cycle_data, metric_column, f" (Cycle {c_int})")
        figure_htmls[f"Cycle {c_int}"] = fig.to_html(full_html=False, include_plotlyjs=False)

    html = wrap_figures_with_cycle_dropdown(figure_htmls, f"Layers Impact — {metric_column}")

    if output_dir:
        path = f"{output_dir}/05_layers_impact.html"
        with open(path, 'w') as f:
            f.write(html)
        print("Saved: 05_layers_impact.html")

    return html
