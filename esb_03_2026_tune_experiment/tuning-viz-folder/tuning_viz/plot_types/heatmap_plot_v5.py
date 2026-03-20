"""
V5 Heatmap: grid (rows=leadtimes, cols=param pairs).
Single file with cycle dropdown. Per-heatmap local color scale.
Subtitle noting best performer shown.
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

from .html_utils_v5 import wrap_figures_with_cycle_dropdown

HEATMAP_PAIRS = [
    ('activation', 'neurons', 'Activation', 'Hidden Units'),
    ('activation', 'num_layers', 'Activation', 'Num Layers'),
    ('num_layers', 'neurons', 'Num Layers', 'Hidden Units'),
]


def _build_heatmap_grid(data: pd.DataFrame, metric_column: str,
                         title_suffix: str = "") -> go.Figure:
    """Build grid: rows=leadtimes, cols=param pairs. Local color scale."""
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

    for row_idx, lt in enumerate(leadtimes, start=1):
        lt_data = clean[clean['leadtime'] == lt]

        for col_idx, (y_col, x_col, y_label, x_label) in enumerate(HEATMAP_PAIRS, start=1):
            pivot = lt_data.pivot_table(
                values=metric_column,
                index=y_col,
                columns=x_col,
                aggfunc='min',
            )

            try:
                pivot = pivot[sorted(pivot.columns, key=lambda x: float(x))]
            except (ValueError, TypeError):
                pass
            try:
                pivot = pivot.reindex(sorted(pivot.index, key=lambda x: float(x)))
            except (ValueError, TypeError):
                pass

            # Local color scale per heatmap
            local_min = np.nanmin(pivot.values)
            local_max = np.nanmax(pivot.values)

            show_colorbar = (col_idx == n_cols)

            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=[str(c) for c in pivot.columns],
                    y=[str(i) for i in pivot.index],
                    colorscale='RdYlGn_r',
                    zmin=local_min,
                    zmax=local_max,
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
        title=dict(
            text=(f"Heatmap Grid: Best {metric_column} per Parameter Combo{title_suffix}"
                  f"<br><sup style='color:#777;'>Each cell shows the best (min) {metric_column} "
                  f"across all configs sharing that parameter pair. Color scale is local per heatmap.</sup>"),
        ),
        height=300 * n_rows + 120,
        width=400 * n_cols + 100,
    )

    return fig


def plot_heatmap_v5(data: pd.DataFrame, metric_column: str,
                     output_dir: Optional[str] = None) -> Optional[str]:
    """
    Single HTML with cycle dropdown. Each view is a leadtime×param-pair grid.
    Local color scales. Subtitle about best performer.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()
    cycles = sorted(clean['cycle'].unique())

    figure_htmls = {}

    # All cycles combined
    fig = _build_heatmap_grid(clean, metric_column, " (All Cycles)")
    figure_htmls["All Cycles"] = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Per cycle
    for cycle in cycles:
        c_int = int(cycle)
        cycle_data = clean[clean['cycle'] == cycle]
        fig = _build_heatmap_grid(cycle_data, metric_column, f" (Cycle {c_int})")
        figure_htmls[f"Cycle {c_int}"] = fig.to_html(full_html=False, include_plotlyjs=False)

    html = wrap_figures_with_cycle_dropdown(figure_htmls, f"Heatmaps — {metric_column}")

    if output_dir:
        path = f"{output_dir}/02_heatmaps.html"
        with open(path, 'w') as f:
            f.write(html)
        print("Saved: 02_heatmaps.html")

    return html
