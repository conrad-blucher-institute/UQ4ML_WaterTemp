"""
V4 Parallel Coordinates: W&B-style overview of all hyperparameters.
Each line = one configuration, colored by metric value.
Axes: activation, num_layers, neurons, leadtime, cycle, metric.
"""

from typing import Optional, List
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def _build_parallel_coords(data: pd.DataFrame, metric_column: str,
                            title_suffix: str = "") -> go.Figure:
    """Build a single parallel coordinates figure."""
    clean = data.copy()

    # Encode activation as numeric for parallel coordinates
    activations = sorted(clean['activation'].unique())
    act_map = {act: i for i, act in enumerate(activations)}
    clean['activation_code'] = clean['activation'].map(act_map)

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
                ),
                dict(
                    label='Hidden Units',
                    values=clean['neurons'],
                ),
                dict(
                    label='Leadtime (h)',
                    values=clean['leadtime'],
                ),
                dict(
                    label='Cycle',
                    values=clean['cycle'],
                ),
                dict(
                    label=metric_column,
                    values=clean[metric_column],
                ),
            ],
        )
    )

    fig.update_layout(
        title=f"Parallel Coordinates: {metric_column}{title_suffix}",
        height=600,
        margin=dict(l=100, r=50, t=80, b=30),
    )

    return fig


def plot_parallel_coords_v4(data: pd.DataFrame, metric_column: str,
                              output_dir: Optional[str] = None) -> List[go.Figure]:
    """
    Parallel coordinates: one combined (all cycles) + one per cycle.
    Each line = one config, colored by metric. Drag axes to filter.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return []

    clean = data.dropna(subset=['leadtime']).copy()
    all_figures = []

    # --- Combined ---
    fig = _build_parallel_coords(clean, metric_column, " (All Cycles)")
    if output_dir:
        fig.write_html(f"{output_dir}/07_parallel_coords_all_cycles.html")
        print("Saved: 07_parallel_coords_all_cycles.html")
    all_figures.append(fig)

    # --- Per cycle ---
    cycles = sorted(clean['cycle'].unique())
    for cycle in cycles:
        c_int = int(cycle)
        cycle_data = clean[clean['cycle'] == cycle]
        fig = _build_parallel_coords(cycle_data, metric_column, f" (Cycle {c_int})")
        if output_dir:
            fig.write_html(f"{output_dir}/07_parallel_coords_cycle_{c_int}.html")
            print(f"Saved: 07_parallel_coords_cycle_{c_int}.html")
        all_figures.append(fig)

    return all_figures
