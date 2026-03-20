"""
V2 Boxplot: metric distribution grouped by each hyperparameter.
One subplot per grouping variable. Flexible sizing with legends.
"""

from typing import Optional
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_boxplot_v2(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[go.Figure]:
    """
    Multiple boxplot rows, each grouped by a different hyperparameter:
    activation, num_layers, leadtime, neurons.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    clean = data.dropna(subset=['leadtime']).copy()

    # Grouping configs: (column, display_label)
    groupings = [
        ('activation', 'Activation Function'),
        ('num_layers', 'Number of Layers'),
        ('leadtime', 'Leadtime (h)'),
        ('neurons', 'Hidden Units'),
    ]

    fig = make_subplots(
        rows=len(groupings), cols=1,
        subplot_titles=[label for _, label in groupings],
        vertical_spacing=0.08,
    )

    colors = px.colors.qualitative.Set2

    for row_idx, (col, label) in enumerate(groupings, start=1):
        unique_vals = sorted(clean[col].unique())
        for i, val in enumerate(unique_vals):
            subset = clean[clean[col] == val]
            fig.add_trace(
                go.Box(
                    y=subset[metric_column],
                    name=str(val),
                    legendgroup=f"{col}_{val}",
                    showlegend=True,
                    boxmean='sd',
                    marker=dict(color=colors[i % len(colors)]),
                ),
                row=row_idx, col=1,
            )
        fig.update_xaxes(title_text=label, row=row_idx, col=1)
        fig.update_yaxes(title_text=metric_column, row=row_idx, col=1)

    fig.update_layout(
        title=f"Performance Distribution by Hyperparameter — {metric_column}",
        height=400 * len(groupings),
        hovermode='closest',
    )

    if output_dir:
        fig.write_html(f"{output_dir}/03_boxplot_v2.html")
        print("Saved: 03_boxplot_v2.html")

    return fig
