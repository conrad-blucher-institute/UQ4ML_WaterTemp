"""
V5 Top Configs: interactive table with separate columns.
Single file with cycle dropdown (All, C1, C2, etc).
Readable header colors.
"""

from typing import Optional
import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .html_utils_v5 import wrap_figures_with_cycle_dropdown


def _build_top_table(data: pd.DataFrame, metric_column: str,
                      top_n: int, title: str) -> go.Figure:
    """Build an interactive table of top N configs with readable headers."""
    top = data.nsmallest(top_n, metric_column).copy()
    top['rank'] = range(1, len(top) + 1)

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
                fill_color='rgb(230, 235, 245)',
                font=dict(color='rgb(30, 30, 30)', size=13),
                align='center',
                height=35,
                line=dict(color='rgb(200, 200, 200)', width=1),
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
                line=dict(color='rgb(220, 220, 220)', width=1),
            ),
        )
    )

    fig.update_layout(
        title=title,
        height=60 + 35 + 28 * len(top),
    )

    return fig


def plot_top_configs_v5(data: pd.DataFrame, metric_column: str,
                         top_n: int = 50, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Single HTML with cycle dropdown. Overall top 50 + per-cycle tables.
    Readable light header with dark text.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()
    cycles = sorted(clean['cycle'].unique())

    figure_htmls = {}

    # All cycles
    fig = _build_top_table(clean, metric_column, top_n,
                            f"Top {top_n} Configurations — {metric_column} (All Cycles)")
    figure_htmls["All Cycles"] = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Per cycle
    for cycle in cycles:
        c_int = int(cycle)
        cycle_data = clean[clean['cycle'] == cycle]
        actual_n = min(top_n, len(cycle_data))
        fig = _build_top_table(cycle_data, metric_column, actual_n,
                                f"Top {actual_n} Configurations — {metric_column} (Cycle {c_int})")
        figure_htmls[f"Cycle {c_int}"] = fig.to_html(full_html=False, include_plotlyjs=False)

    html = wrap_figures_with_cycle_dropdown(figure_htmls, f"Top Configurations — {metric_column}")

    if output_dir:
        path = f"{output_dir}/04_top_configs.html"
        with open(path, 'w') as f:
            f.write(html)
        print("Saved: 04_top_configs.html")

    return html
