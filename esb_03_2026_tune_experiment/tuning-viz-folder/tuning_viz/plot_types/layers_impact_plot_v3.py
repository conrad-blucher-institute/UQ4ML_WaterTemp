"""
V3 Layers Impact: num_layers on x-axis, one boxplot per activation function.
val_mae always on y-axis. Categorical x. Combined + per-cycle views.
"""

from typing import Optional, List
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


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


def plot_layers_impact_v3(data: pd.DataFrame, metric_column: str,
                           output_dir: Optional[str] = None) -> List[go.Figure]:
    """
    Layers impact: combined (all cycles) + per-cycle views.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return []

    clean = data.dropna(subset=['leadtime', 'num_layers']).copy()
    all_figures = []

    # --- Combined ---
    fig = _build_layers_figure(clean, metric_column, " (All Cycles)")
    if output_dir:
        fig.write_html(f"{output_dir}/05_layers_impact_all_cycles.html")
        print("Saved: 05_layers_impact_all_cycles.html")
    all_figures.append(fig)

    # --- Per cycle ---
    cycles = sorted(clean['cycle'].unique())
    for cycle in cycles:
        c_int = int(cycle)
        cycle_data = clean[clean['cycle'] == cycle]
        fig = _build_layers_figure(cycle_data, metric_column, f" (Cycle {c_int})")
        if output_dir:
            fig.write_html(f"{output_dir}/05_layers_impact_cycle_{c_int}.html")
            print(f"Saved: 05_layers_impact_cycle_{c_int}.html")
        all_figures.append(fig)

    return all_figures
