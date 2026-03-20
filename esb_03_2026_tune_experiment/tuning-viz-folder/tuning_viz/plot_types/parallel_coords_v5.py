"""
V5 Parallel Coordinates: single combined plot only (no per-cycle files).
Axis tick marks show only actual parameter values.
"""

from typing import Optional
import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_parallel_coords_v5(data: pd.DataFrame, metric_column: str,
                              output_dir: Optional[str] = None) -> Optional[go.Figure]:
    """
    Single parallel coordinates plot with all data.
    Cycle is an axis (use the built-in drag-to-filter on it).
    All axes show only actual unique values as tick marks.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()

    # Encode activation as numeric
    activations = sorted(clean['activation'].unique())
    act_map = {act: i for i, act in enumerate(activations)}
    clean['activation_code'] = clean['activation'].map(act_map)

    # Get unique sorted values for tick marks
    unique_layers = sorted(clean['num_layers'].unique())
    unique_neurons = sorted(clean['neurons'].unique())
    unique_leadtimes = sorted(clean['leadtime'].unique())
    unique_cycles = sorted(clean['cycle'].unique())

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=clean[metric_column],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title=metric_column),
            ),
            dimensions=[
                dict(
                    label='Activation',
                    values=clean['activation_code'],
                    tickvals=list(act_map.values()),
                    ticktext=list(act_map.keys()),
                ),
                dict(
                    label='Num Layers',
                    values=clean['num_layers'],
                    tickvals=[int(v) for v in unique_layers],
                    ticktext=[str(int(v)) for v in unique_layers],
                ),
                dict(
                    label='Hidden Units',
                    values=clean['neurons'],
                    tickvals=[int(v) for v in unique_neurons],
                    ticktext=[str(int(v)) for v in unique_neurons],
                ),
                dict(
                    label='Leadtime (h)',
                    values=clean['leadtime'],
                    tickvals=[int(v) for v in unique_leadtimes],
                    ticktext=[str(int(v)) for v in unique_leadtimes],
                ),
                dict(
                    label='Cycle',
                    values=clean['cycle'],
                    tickvals=[int(v) for v in unique_cycles],
                    ticktext=[str(int(v)) for v in unique_cycles],
                ),
                dict(
                    label=metric_column,
                    values=clean[metric_column],
                ),
            ],
        )
    )

    fig.update_layout(
        title=f"Parallel Coordinates: {metric_column} (All Data)",
        height=600,
        margin=dict(l=100, r=50, t=80, b=30),
    )

    if output_dir:
        fig.write_html(f"{output_dir}/07_parallel_coords.html")
        print("Saved: 07_parallel_coords.html")

    return fig
