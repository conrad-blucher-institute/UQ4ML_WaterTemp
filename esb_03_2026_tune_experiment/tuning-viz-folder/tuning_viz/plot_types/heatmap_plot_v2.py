"""
V2 Heatmap: all leadtimes as subplots in a single figure.
Activation vs neurons, colored by best metric. Flexible sizing.
"""

from typing import Optional
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_heatmap_v2(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[go.Figure]:
    """
    Single figure with one heatmap subplot per leadtime.
    Activation (y) vs neurons (x), colored by best (min) metric value.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    clean = data.dropna(subset=['leadtime']).copy()
    leadtimes = sorted(clean['leadtime'].unique())

    fig = make_subplots(
        rows=1, cols=len(leadtimes),
        subplot_titles=[f"Leadtime: {int(lt)}h" for lt in leadtimes],
        horizontal_spacing=0.06,
    )

    # Compute global min/max for consistent color scale
    global_min = None
    global_max = None
    pivots = []
    for lt in leadtimes:
        subset = clean[clean['leadtime'] == lt]
        pivot = subset.pivot_table(
            values=metric_column,
            index='activation',
            columns='neurons',
            aggfunc='min',
        )
        pivots.append(pivot)
        if global_min is None:
            global_min = pivot.min().min()
            global_max = pivot.max().max()
        else:
            global_min = min(global_min, pivot.min().min())
            global_max = max(global_max, pivot.max().max())

    for col_idx, (lt, pivot) in enumerate(zip(leadtimes, pivots), start=1):
        show_colorbar = col_idx == len(leadtimes)
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='RdYlGn_r',
                zmin=global_min,
                zmax=global_max,
                text=np.round(pivot.values, 4),
                texttemplate='%{text:.3f}',
                textfont={"size": 9},
                showscale=show_colorbar,
                colorbar=dict(title=metric_column) if show_colorbar else None,
            ),
            row=1, col=col_idx,
        )
        fig.update_xaxes(title_text="Hidden Units", row=1, col=col_idx)
        if col_idx == 1:
            fig.update_yaxes(title_text="Activation Function", row=1, col=col_idx)

    fig.update_layout(
        title=f"Heatmap: Activation vs Neurons — Best {metric_column} (all leadtimes)",
        height=400 + 30 * len(pivots[0].index) if pivots else 500,
    )

    if output_dir:
        fig.write_html(f"{output_dir}/02_heatmap_v2.html")
        print("Saved: 02_heatmap_v2.html")

    return fig
