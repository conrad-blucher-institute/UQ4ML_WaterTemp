"""
V4 Heatmap: grid layout — rows = leadtimes, cols = parameter pairs.
Leadtime is NOT a heatmap parameter; it separates the rows.
Parameter pairs: activation×neurons, activation×layers, layers×neurons.
Combined + per-cycle views.
"""

from typing import Optional, List
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Parameter pairs: (y_col, x_col, y_label, x_label)
HEATMAP_PAIRS = [
    ('activation', 'neurons', 'Activation', 'Hidden Units'),
    ('activation', 'num_layers', 'Activation', 'Num Layers'),
    ('num_layers', 'neurons', 'Num Layers', 'Hidden Units'),
]


def _build_heatmap_grid(data: pd.DataFrame, metric_column: str,
                         title_suffix: str = "") -> go.Figure:
    """Build a grid of heatmaps: rows=leadtimes, cols=param pairs."""
    clean = data.dropna(subset=['leadtime']).copy()
    leadtimes = sorted(clean['leadtime'].unique())

    n_rows = len(leadtimes)
    n_cols = len(HEATMAP_PAIRS)

    subplot_titles = []
    for lt in leadtimes:
        for _, _, y_label, x_label in HEATMAP_PAIRS:
            subplot_titles.append(f"{y_label} vs {x_label} — LT {int(lt)}h")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
    )

    # Compute global min/max for consistent color scale
    global_min = clean[metric_column].min()
    global_max = clean[metric_column].max()

    for row_idx, lt in enumerate(leadtimes, start=1):
        lt_data = clean[clean['leadtime'] == lt]

        for col_idx, (y_col, x_col, y_label, x_label) in enumerate(HEATMAP_PAIRS, start=1):
            pivot = lt_data.pivot_table(
                values=metric_column,
                index=y_col,
                columns=x_col,
                aggfunc='min',
            )

            # Sort numerically where possible
            try:
                pivot = pivot[sorted(pivot.columns, key=lambda x: float(x))]
            except (ValueError, TypeError):
                pass
            try:
                pivot = pivot.reindex(sorted(pivot.index, key=lambda x: float(x)))
            except (ValueError, TypeError):
                pass

            show_colorbar = (row_idx == 1 and col_idx == n_cols)

            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=[str(c) for c in pivot.columns],
                    y=[str(i) for i in pivot.index],
                    colorscale='RdYlGn_r',
                    zmin=global_min,
                    zmax=global_max,
                    text=np.round(pivot.values, 4),
                    texttemplate='%{text:.3f}',
                    textfont={"size": 8},
                    showscale=show_colorbar,
                    colorbar=dict(title=metric_column) if show_colorbar else None,
                ),
                row=row_idx, col=col_idx,
            )

            fig.update_xaxes(
                title_text=x_label if row_idx == n_rows else "",
                row=row_idx, col=col_idx,
                type='category',
            )
            fig.update_yaxes(
                title_text=y_label if col_idx == 1 else "",
                row=row_idx, col=col_idx,
                type='category',
            )

    fig.update_layout(
        title=f"Heatmap Grid: Best {metric_column} by Leadtime{title_suffix}",
        height=300 * n_rows + 80,
        width=400 * n_cols + 100,
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


def plot_heatmap_v4(data: pd.DataFrame, metric_column: str,
                     output_dir: Optional[str] = None) -> List[go.Figure]:
    """
    Generate heatmap grids: one combined (all cycles) + one per cycle.
    Each grid: rows=leadtimes, cols=param pairs.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return []

    clean = data.dropna(subset=['leadtime']).copy()
    all_figures = []

    # --- Combined (all cycles) ---
    fig = _build_heatmap_grid(clean, metric_column, title_suffix=" (All Cycles)")
    if output_dir:
        fig.write_html(f"{output_dir}/02_heatmaps_all_cycles.html")
        print("Saved: 02_heatmaps_all_cycles.html")
    all_figures.append(fig)

    # --- Per cycle ---
    cycles = sorted(clean['cycle'].unique())
    cycle_figs = []
    for cycle in cycles:
        c_int = int(cycle)
        cycle_data = clean[clean['cycle'] == cycle]
        fig = _build_heatmap_grid(cycle_data, metric_column, title_suffix=f" (Cycle {c_int})")
        cycle_figs.append(fig)
        all_figures.append(fig)

    if output_dir and cycle_figs:
        html = _stitch_figures_to_html(cycle_figs, f"Heatmap Grids per Cycle — {metric_column}")
        with open(f"{output_dir}/02_heatmaps_per_cycle.html", 'w') as f:
            f.write(html)
        print("Saved: 02_heatmaps_per_cycle.html")

    return all_figures
