"""
V5 Boxplot: one figure per hyperparameter grouping, stitched together.
Single file with cycle dropdown. Independent legends per subplot.
Categorical x-axes.
"""

from typing import Optional
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .html_utils_v5 import wrap_figures_with_cycle_dropdown

GROUPINGS = [
    ('activation', 'Activation Function'),
    ('num_layers', 'Number of Layers'),
    ('leadtime', 'Leadtime (h)'),
    ('neurons', 'Hidden Units'),
]


def _build_single_boxplot(data: pd.DataFrame, metric_column: str,
                           group_col: str, group_label: str,
                           title_suffix: str = "") -> go.Figure:
    """Build a single boxplot figure for one grouping variable."""
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    unique_vals = data[group_col].unique()
    try:
        unique_vals = sorted(unique_vals, key=lambda x: float(x))
    except (ValueError, TypeError):
        unique_vals = sorted(unique_vals)

    for i, val in enumerate(unique_vals):
        subset = data[data[group_col] == val]
        fig.add_trace(
            go.Box(
                y=subset[metric_column],
                name=str(val) if group_col == 'activation' else str(int(val)),
                boxmean='sd',
                marker=dict(color=colors[i % len(colors)]),
            )
        )

    fig.update_layout(
        title=f"{metric_column} by {group_label}{title_suffix}",
        xaxis_title=group_label,
        yaxis_title=metric_column,
        xaxis=dict(type='category'),
        height=450,
        hovermode='closest',
    )

    return fig


def _stitch_boxplots(data: pd.DataFrame, metric_column: str,
                      title_suffix: str = "") -> str:
    """Build all grouping boxplots and stitch into one HTML fragment."""
    parts = []
    first = True
    for group_col, group_label in GROUPINGS:
        fig = _build_single_boxplot(data, metric_column, group_col, group_label, title_suffix)
        parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn' if first else False))
        parts.append("<hr style='margin: 20px 0;'>")
        first = False
    return "\n".join(parts)


def plot_boxplot_v5(data: pd.DataFrame, metric_column: str,
                     output_dir: Optional[str] = None) -> Optional[str]:
    """
    Single HTML with cycle dropdown. Each view stitches boxplots for
    all hyperparameter groupings with independent legends.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()
    cycles = sorted(clean['cycle'].unique())

    figure_htmls = {}

    # All cycles
    figure_htmls["All Cycles"] = _stitch_boxplots(clean, metric_column, " (All Cycles)")

    # Per cycle
    for cycle in cycles:
        c_int = int(cycle)
        cycle_data = clean[clean['cycle'] == cycle]
        figure_htmls[f"Cycle {c_int}"] = _stitch_boxplots(cycle_data, metric_column, f" (Cycle {c_int})")

    html = wrap_figures_with_cycle_dropdown(figure_htmls, f"Boxplots — {metric_column}")

    if output_dir:
        path = f"{output_dir}/03_boxplots.html"
        with open(path, 'w') as f:
            f.write(html)
        print("Saved: 03_boxplots.html")

    return html
