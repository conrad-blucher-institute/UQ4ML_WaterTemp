"""
V3 Top Configs: interactive tables with individual columns.
Overall top 50 + per-cycle top tables. Stitched into HTML.
"""

from typing import Optional, List
import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def _build_top_table(data: pd.DataFrame, metric_column: str,
                      top_n: int, title: str) -> go.Figure:
    """Build an interactive table of top N configs."""
    top = data.nsmallest(top_n, metric_column).copy()
    top['rank'] = range(1, len(top) + 1)

    # Color rows by metric
    vals = top[metric_column].values
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    row_colors = [
        f"rgba({int(200 * n)}, {int(200 * (1 - n))}, 80, 0.25)"
        for n in norm
    ]
    white = ['white'] * len(top)

    fig = go.Figure(
        data=go.Table(
            header=dict(
                values=[
                    '<b>Rank</b>',
                    '<b>Activation</b>',
                    '<b>Layers</b>',
                    '<b>Neurons</b>',
                    '<b>Leadtime</b>',
                    '<b>Cycle</b>',
                    f'<b>{metric_column}</b>',
                ],
                fill_color='rgb(55, 55, 55)',
                font=dict(color='white', size=13),
                align='center',
                height=35,
            ),
            cells=dict(
                values=[
                    top['rank'].tolist(),
                    top['activation'].tolist(),
                    top['num_layers'].astype(int).tolist(),
                    top['neurons'].astype(int).tolist(),
                    [f"{int(v)}h" for v in top['leadtime']],
                    top['cycle'].astype(int).tolist(),
                    [f"{v:.4f}" for v in top[metric_column]],
                ],
                fill_color=[
                    white,
                    row_colors,
                    row_colors,
                    row_colors,
                    row_colors,
                    row_colors,
                    row_colors,
                ],
                font=dict(size=12),
                align='center',
                height=28,
            ),
        )
    )

    fig.update_layout(
        title=title,
        height=60 + 35 + 28 * len(top),
    )

    return fig


def _stitch_figures_to_html(figures: List[go.Figure], page_title: str) -> str:
    """Combine multiple plotly figures into one HTML."""
    parts = [
        f"<html><head><title>{page_title}</title></head><body>",
        f"<h2 style='font-family: sans-serif; text-align: center;'>{page_title}</h2>",
    ]
    for fig in figures:
        parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        parts.append("<hr style='margin: 30px 0;'>")
    parts.append("</body></html>")
    return "\n".join(parts)


def plot_top_configs_v3(data: pd.DataFrame, metric_column: str,
                         top_n: int = 50, output_dir: Optional[str] = None) -> List[go.Figure]:
    """
    Top configs tables: overall top 50 + per-cycle top tables.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return []

    clean = data.dropna(subset=['leadtime']).copy()
    all_figures = []

    # --- Overall top N ---
    overall_fig = _build_top_table(clean, metric_column, top_n,
                                    f"Top {top_n} Configurations — {metric_column} (All Cycles)")
    all_figures.append(overall_fig)

    if output_dir:
        overall_fig.write_html(f"{output_dir}/04_top_configs_overall.html")
        print("Saved: 04_top_configs_overall.html")

    # --- Per cycle ---
    cycles = sorted(clean['cycle'].unique())
    per_cycle_figs = []
    for cycle in cycles:
        c_int = int(cycle)
        cycle_data = clean[clean['cycle'] == cycle]
        actual_n = min(top_n, len(cycle_data))
        fig = _build_top_table(cycle_data, metric_column, actual_n,
                                f"Top {actual_n} Configurations — {metric_column} (Cycle {c_int})")
        per_cycle_figs.append(fig)
        all_figures.append(fig)

    if output_dir:
        html = _stitch_figures_to_html(per_cycle_figs,
                                        f"Top Configurations per Cycle — {metric_column}")
        with open(f"{output_dir}/04_top_configs_per_cycle.html", 'w') as f:
            f.write(html)
        print("Saved: 04_top_configs_per_cycle.html")

    return all_figures
