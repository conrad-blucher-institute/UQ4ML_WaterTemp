"""
V7 Top Configs: top 50 with cycle dropdown + top 10 per cycle with All Cycles.
Dark header with white text for readability.
"""

from typing import Optional
import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .html_utils_v7 import wrap_figures_with_cycle_dropdown


def _build_top_table(data: pd.DataFrame, metric_column: str,
                      top_n: int, title: str) -> go.Figure:
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
                    '<b>Rank</b>', '<b>Activation</b>', '<b>Layers</b>',
                    '<b>Neurons</b>', '<b>Leadtime</b>', '<b>Cycle</b>',
                    f'<b>{metric_column}</b>',
                ],
                fill_color='rgb(50, 60, 80)',
                font=dict(color='white', size=13),
                align='center',
                height=35,
                line=dict(color='rgb(40, 50, 70)', width=1),
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
                fill_color=[white, row_colors, row_colors, row_colors,
                            row_colors, row_colors, row_colors],
                font=dict(size=12),
                align='center',
                height=28,
                line=dict(color='rgb(220, 220, 220)', width=1),
            ),
        )
    )

    fig.update_layout(title=title, height=60 + 35 + 28 * len(top))
    return fig


def plot_top_configs_v7(data: pd.DataFrame, metric_column: str,
                         top_n: int = 50, output_dir: Optional[str] = None) -> Optional[str]:
    """Top 50 table with cycle dropdown."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()
    cycles = sorted(clean['cycle'].unique())

    figure_htmls = {}
    fig = _build_top_table(clean, metric_column, top_n,
                            f"Top {top_n} — {metric_column} (All Cycles)")
    figure_htmls["All Cycles"] = fig.to_html(full_html=False, include_plotlyjs='cdn')

    for cycle in cycles:
        c_int = int(cycle)
        cycle_data = clean[clean['cycle'] == cycle]
        actual_n = min(top_n, len(cycle_data))
        fig = _build_top_table(cycle_data, metric_column, actual_n,
                                f"Top {actual_n} — {metric_column} (Cycle {c_int})")
        figure_htmls[f"Cycle {c_int}"] = fig.to_html(full_html=False, include_plotlyjs=False)

    html = wrap_figures_with_cycle_dropdown(figure_htmls, f"Top Configurations — {metric_column}")

    if output_dir:
        with open(f"{output_dir}/04_top_configs.html", 'w') as f:
            f.write(html)
        print("Saved: 04_top_configs.html")
    return html


def plot_top10_per_cycle_v7(data: pd.DataFrame, metric_column: str,
                              output_dir: Optional[str] = None) -> Optional[str]:
    """Top 10 per cycle + All Cycles, shown as wrapping flex grid."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()
    cycles = sorted(clean['cycle'].unique())

    # Build cells: All Cycles first, then per cycle
    cells = []

    # All Cycles
    actual_n = min(10, len(clean))
    fig = _build_top_table(clean, metric_column, actual_n,
                            f"All Cycles — Top {actual_n}")
    fig.update_layout(height=60 + 35 + 28 * actual_n, margin=dict(t=40, b=10, l=10, r=10))
    cells.append(('All Cycles', fig))

    for cycle in cycles:
        c_int = int(cycle)
        cycle_data = clean[clean['cycle'] == cycle]
        actual_n = min(10, len(cycle_data))
        fig = _build_top_table(cycle_data, metric_column, actual_n,
                                f"Cycle {c_int} — Top {actual_n}")
        fig.update_layout(height=60 + 35 + 28 * actual_n, margin=dict(t=40, b=10, l=10, r=10))
        cells.append((f'Cycle {c_int}', fig))

    parts = [
        f"""<!DOCTYPE html>
<html><head>
<title>Top 10 per Cycle — {metric_column}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           margin: 0; padding: 20px; }}
    h2 {{ text-align: center; font-size: 18px; margin-bottom: 16px; }}
    .grid {{ display: flex; flex-wrap: wrap; gap: 16px; justify-content: center; }}
    .cell {{ flex: 1 1 400px; max-width: 600px; }}
</style>
</head><body>
<h2>Top 10 Configurations per Cycle — {metric_column}</h2>
<div class="grid">"""
    ]

    for i, (label, fig) in enumerate(cells):
        js_flag = 'cdn' if i == 0 else False
        parts.append(f'<div class="cell">{fig.to_html(full_html=False, include_plotlyjs=js_flag)}</div>')

    parts.append("</div></body></html>")
    html = "\n".join(parts)

    if output_dir:
        with open(f"{output_dir}/04_top10_per_cycle.html", 'w') as f:
            f.write(html)
        print("Saved: 04_top10_per_cycle.html")
    return html
