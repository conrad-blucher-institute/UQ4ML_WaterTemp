"""
V3 Boxplot: one standalone figure per hyperparameter grouping,
stitched into a single HTML so each has its own independent legend.
Combined + per-cycle views. Categorical x-axes.
"""

from typing import Optional, List
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# (column, display_label)
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

    # Sort categories numerically if possible, else alphabetically
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


def _stitch_figures_to_html(figures: List[go.Figure], page_title: str) -> str:
    """Combine multiple plotly figures into a single HTML string."""
    parts = [
        f"<html><head><title>{page_title}</title></head><body>",
        f"<h2 style='font-family: sans-serif; text-align: center;'>{page_title}</h2>",
    ]
    for fig in figures:
        parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        parts.append("<hr style='margin: 30px 0;'>")
    parts.append("</body></html>")
    return "\n".join(parts)


def plot_boxplot_v3(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> List[go.Figure]:
    """
    Generate boxplots for each hyperparameter grouping.
    Each grouping is its own figure with independent legend.
    Stitched into single HTML files. Combined + per-cycle.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return []

    clean = data.dropna(subset=['leadtime']).copy()
    all_figures = []

    # --- Combined (all cycles) ---
    combined_figs = []
    for group_col, group_label in GROUPINGS:
        fig = _build_single_boxplot(clean, metric_column, group_col, group_label, " (All Cycles)")
        combined_figs.append(fig)
        all_figures.append(fig)

    if output_dir:
        html = _stitch_figures_to_html(combined_figs, f"Boxplots by Hyperparameter — {metric_column} (All Cycles)")
        with open(f"{output_dir}/03_boxplots_all_cycles.html", 'w') as f:
            f.write(html)
        print("Saved: 03_boxplots_all_cycles.html")

    # --- Per cycle ---
    cycles = sorted(clean['cycle'].unique())
    for cycle in cycles:
        c_int = int(cycle)
        cycle_data = clean[clean['cycle'] == cycle]
        cycle_figs = []

        for group_col, group_label in GROUPINGS:
            fig = _build_single_boxplot(cycle_data, metric_column, group_col, group_label, f" (Cycle {c_int})")
            cycle_figs.append(fig)
            all_figures.append(fig)

        if output_dir:
            html = _stitch_figures_to_html(cycle_figs, f"Boxplots by Hyperparameter — {metric_column} (Cycle {c_int})")
            with open(f"{output_dir}/03_boxplots_cycle_{c_int}.html", 'w') as f:
                f.write(html)
            print(f"Saved: 03_boxplots_cycle_{c_int}.html")

    return all_figures
