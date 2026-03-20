"""
V8 Top Configs: top 50 with cycle dropdown (Plotly table).
Top 10 per cycle uses plain HTML tables — no column reordering, clear headers.
Dark header with white text.
"""

from typing import Optional
import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .html_utils_v8 import wrap_figures_with_cycle_dropdown


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
                values=['<b>Rank</b>', '<b>Activation</b>', '<b>Layers</b>',
                        '<b>Neurons</b>', '<b>Leadtime</b>', '<b>Cycle</b>',
                        f'<b>{metric_column}</b>'],
                fill_color='rgb(50, 60, 80)',
                font=dict(color='white', size=13),
                align='center', height=35,
                line=dict(color='rgb(40, 50, 70)', width=1),
            ),
            cells=dict(
                values=[
                    top['rank'].tolist(), top['activation'].tolist(),
                    top['num_layers'].astype(int).tolist(),
                    top['neurons'].astype(int).tolist(),
                    [f"{int(v)}h" for v in top['leadtime']],
                    top['cycle'].astype(int).tolist(),
                    [f"{v:.4f}" for v in top[metric_column]],
                ],
                fill_color=[white, row_colors, row_colors, row_colors,
                            row_colors, row_colors, row_colors],
                font=dict(size=12), align='center', height=28,
                line=dict(color='rgb(220, 220, 220)', width=1),
            ),
        )
    )
    fig.update_layout(title=title, height=60 + 35 + 28 * len(top))
    return fig


def plot_top_configs_v8(data: pd.DataFrame, metric_column: str,
                         top_n: int = 50, output_dir: Optional[str] = None) -> Optional[str]:
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


def _html_table(data: pd.DataFrame, metric_column: str, top_n: int, title: str) -> str:
    """Build a plain HTML table — no Plotly, no column dragging."""
    top = data.nsmallest(top_n, metric_column).copy()

    rows_html = ""
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        val = row[metric_column]
        rows_html += f"""<tr>
            <td>{rank}</td><td>{row['activation']}</td><td>{int(row['num_layers'])}</td>
            <td>{int(row['neurons'])}</td><td>{int(row['leadtime'])}h</td>
            <td>{int(row['cycle'])}</td><td>{val:.4f}</td>
        </tr>\n"""

    return f"""<div class="table-card">
        <h3>{title}</h3>
        <table>
            <thead><tr>
                <th>Rank</th><th>Activation</th><th>Layers</th>
                <th>Neurons</th><th>Leadtime</th><th>Cycle</th><th>{metric_column}</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>"""


def plot_top10_per_cycle_v8(data: pd.DataFrame, metric_column: str,
                              output_dir: Optional[str] = None) -> Optional[str]:
    """Top 10 per cycle + All Cycles, plain HTML tables in flex grid."""
    clean = data.dropna(subset=['leadtime']).copy()
    cycles = sorted(clean['cycle'].unique())

    tables_html = _html_table(clean, metric_column, 10, "All Cycles — Top 10")

    for cycle in cycles:
        c_int = int(cycle)
        cycle_data = clean[clean['cycle'] == cycle]
        actual_n = min(10, len(cycle_data))
        tables_html += _html_table(cycle_data, metric_column, actual_n, f"Cycle {c_int} — Top {actual_n}")

    html = f"""<!DOCTYPE html>
<html><head>
<title>Top 10 per Cycle — {metric_column}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           margin: 0; padding: 20px; background: #fafafa; }}
    h2 {{ text-align: center; font-size: 18px; margin-bottom: 16px; }}
    .grid {{ display: flex; flex-wrap: wrap; gap: 16px; justify-content: center; }}
    .table-card {{ flex: 1 1 380px; max-width: 550px; background: white;
                   border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);
                   padding: 12px; }}
    .table-card h3 {{ margin: 0 0 8px 0; font-size: 14px; color: #333; text-align: center; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th {{ background: rgb(50, 60, 80); color: white; padding: 8px 6px;
         font-weight: 600; text-align: center; white-space: nowrap; }}
    td {{ padding: 6px; text-align: center; border-bottom: 1px solid #eee; }}
    tr:hover {{ background: #f0f7f7; }}
</style>
</head><body>
<h2>Top 10 Configurations per Cycle — {metric_column}</h2>
<div class="grid">
{tables_html}
</div>
</body></html>"""

    if output_dir:
        with open(f"{output_dir}/04_top10_per_cycle.html", 'w') as f:
            f.write(html)
        print("Saved: 04_top10_per_cycle.html")
    return html
