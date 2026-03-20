"""
V2 Scatter plots: one plot per hyperparameter x-axis, all leadtimes combined.
Color = activation function, symbol = leadtime (toggleable in legend).
Flexible sizing. Filters out NaN leadtimes.
"""

from typing import Optional
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_scatter_v2(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[go.Figure]:
    """
    Multi-row scatter: each row uses a different hyperparameter as x-axis.
    All leadtimes on same plot, toggleable via legend.
    Color = activation, symbol = leadtime.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    # Filter out NaN leadtimes
    clean = data.dropna(subset=['leadtime']).copy()
    clean['leadtime_label'] = clean['leadtime'].astype(int).astype(str) + 'h'

    # Define x-axis configs: (column_name, display_label)
    x_axes = [
        ('neurons', 'Hidden Units'),
        ('num_layers', 'Number of Layers'),
    ]

    activations = sorted(clean['activation'].unique())
    leadtime_labels = sorted(clean['leadtime_label'].unique(), key=lambda x: int(x.replace('h', '')))

    # Color for activation, symbol for leadtime
    act_colors = px.colors.qualitative.Set1
    act_color_map = {act: act_colors[i % len(act_colors)] for i, act in enumerate(activations)}

    symbol_list = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'star', 'hexagon']
    lt_symbol_map = {lt: symbol_list[i % len(symbol_list)] for i, lt in enumerate(leadtime_labels)}

    fig = make_subplots(
        rows=len(x_axes), cols=1,
        subplot_titles=[label for _, label in x_axes],
        vertical_spacing=0.12,
    )

    # Track which legend entries we've already added (avoid duplicates across rows)
    seen_legend = set()

    for row_idx, (x_col, x_label) in enumerate(x_axes, start=1):
        for activation in activations:
            for lt_label in leadtime_labels:
                subset = clean[(clean['activation'] == activation) & (clean['leadtime_label'] == lt_label)]
                if subset.empty:
                    continue

                legend_name = f"{activation} | {lt_label}"
                show = legend_name not in seen_legend
                seen_legend.add(legend_name)

                fig.add_trace(
                    go.Scatter(
                        x=subset[x_col],
                        y=subset[metric_column],
                        mode='markers',
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=show,
                        marker=dict(
                            size=7,
                            color=act_color_map[activation],
                            symbol=lt_symbol_map[lt_label],
                            opacity=0.7,
                            line=dict(width=0.5, color='white'),
                        ),
                        text=[
                            f"Activation: {a}<br>Neurons: {n}<br>Layers: {l}<br>"
                            f"Leadtime: {lt}<br>{metric_column}: {m:.4f}"
                            for a, n, l, lt, m in zip(
                                subset['activation'],
                                subset['neurons'],
                                subset['num_layers'],
                                subset['leadtime_label'],
                                subset[metric_column],
                            )
                        ],
                        hoverinfo='text',
                    ),
                    row=row_idx, col=1,
                )

        fig.update_xaxes(title_text=x_label, row=row_idx, col=1)
        fig.update_yaxes(title_text=metric_column, row=row_idx, col=1)

    fig.update_layout(
        title=f"Hyperparameter Scatter: {metric_column}",
        height=450 * len(x_axes),
        hovermode='closest',
        legend=dict(
            title="Activation | Leadtime",
            itemsizing='constant',
        ),
    )

    if output_dir:
        fig.write_html(f"{output_dir}/01_scatter_v2.html")
        print("Saved: 01_scatter_v2.html")

    return fig
