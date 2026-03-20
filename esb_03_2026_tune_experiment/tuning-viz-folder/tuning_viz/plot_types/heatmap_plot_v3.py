"""
V3 Heatmap: all pairwise hyperparameter combinations.
Each pair gets its own heatmap. Combined + per-cycle views.
Stitched into single HTML files.
"""

from typing import Optional, List
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Hyperparameter pairs to compare: (y_col, x_col, y_label, x_label)
HEATMAP_PAIRS = [
    ('activation', 'neurons', 'Activation', 'Hidden Units'),
    ('activation', 'num_layers', 'Activation', 'Num Layers'),
    ('activation', 'leadtime', 'Activation', 'Leadtime'),
    ('num_layers', 'neurons', 'Num Layers', 'Hidden Units'),
    ('num_layers', 'leadtime', 'Num Layers', 'Leadtime'),
    ('neurons', 'leadtime', 'Hidden Units', 'Leadtime'),
]


def _build_single_heatmap(data: pd.DataFrame, metric_column: str,
                           y_col: str, x_col: str,
                           y_label: str, x_label: str,
                           title: str) -> go.Figure:
    """Build one heatmap for a given pair of hyperparameters."""
    pivot = data.pivot_table(
        values=metric_column,
        index=y_col,
        columns=x_col,
        aggfunc='min',
    )

    # Sort columns/index numerically if possible
    try:
        pivot = pivot[sorted(pivot.columns, key=lambda x: float(x))]
    except (ValueError, TypeError):
        pass
    try:
        pivot = pivot.reindex(sorted(pivot.index, key=lambda x: float(x)))
    except (ValueError, TypeError):
        pass

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            colorscale='RdYlGn_r',
            text=np.round(pivot.values, 4),
            texttemplate='%{text:.3f}',
            textfont={"size": 10},
            colorbar=dict(title=metric_column),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=350 + 25 * len(pivot.index),
        xaxis=dict(type='category'),
        yaxis=dict(type='category'),
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


def plot_heatmap_v3(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> List[go.Figure]:
    """
    Generate heatmaps for all pairwise hyperparameter combos.
    Combined (all cycles) + per-cycle views.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return []

    clean = data.dropna(subset=['leadtime']).copy()
    all_figures = []

    # --- Combined (all cycles) ---
    combined_figs = []
    for y_col, x_col, y_label, x_label in HEATMAP_PAIRS:
        title = f"{y_label} vs {x_label} — Best {metric_column} (All Cycles)"
        fig = _build_single_heatmap(clean, metric_column, y_col, x_col, y_label, x_label, title)
        combined_figs.append(fig)
        all_figures.append(fig)

    if output_dir:
        html = _stitch_figures_to_html(combined_figs, f"Heatmaps: All Pairwise Combos — {metric_column} (All Cycles)")
        with open(f"{output_dir}/02_heatmaps_all_cycles.html", 'w') as f:
            f.write(html)
        print("Saved: 02_heatmaps_all_cycles.html")

    # --- Per cycle ---
    cycles = sorted(clean['cycle'].unique())
    for cycle in cycles:
        c_int = int(cycle)
        cycle_data = clean[clean['cycle'] == cycle]
        cycle_figs = []

        for y_col, x_col, y_label, x_label in HEATMAP_PAIRS:
            title = f"{y_label} vs {x_label} — Best {metric_column} (Cycle {c_int})"
            fig = _build_single_heatmap(cycle_data, metric_column, y_col, x_col, y_label, x_label, title)
            cycle_figs.append(fig)
            all_figures.append(fig)

        if output_dir:
            html = _stitch_figures_to_html(cycle_figs, f"Heatmaps: All Pairwise Combos — {metric_column} (Cycle {c_int})")
            with open(f"{output_dir}/02_heatmaps_cycle_{c_int}.html", 'w') as f:
                f.write(html)
            print(f"Saved: 02_heatmaps_cycle_{c_int}.html")

    return all_figures
