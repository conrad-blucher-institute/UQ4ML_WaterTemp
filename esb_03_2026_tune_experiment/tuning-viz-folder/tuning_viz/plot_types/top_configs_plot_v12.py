"""
V12 Top Configs: top 50 with cycle + leadtime dropdowns.
Top 10 per cycle uses plain HTML tables with leadtime filter.
"""

from typing import Optional
import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


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


def _make_combo_key(cycle, leadtime):
    """Create a unique key for cycle+leadtime combination."""
    c = "All" if cycle is None else f"C{int(cycle)}"
    l = "All" if leadtime is None else f"L{int(leadtime)}"
    return f"{c}_{l}"


def plot_top_configs_v12(data: pd.DataFrame, metric_column: str,
                          top_n: int = 50, output_dir: Optional[str] = None) -> Optional[str]:
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()
    cycles = sorted(clean['cycle'].unique())
    leadtimes = sorted(clean['leadtime'].unique())

    # Build a figure for every (cycle, leadtime) combination
    figure_htmls = {}
    combos = [(None, None)]  # All/All
    combos += [(None, lt) for lt in leadtimes]  # All cycles, specific leadtime
    combos += [(cy, None) for cy in cycles]  # Specific cycle, all leadtimes
    combos += [(cy, lt) for cy in cycles for lt in leadtimes]

    first = True
    for cycle, leadtime in combos:
        subset = clean.copy()
        c_label = "All Cycles"
        l_label = "All Leadtimes"
        if cycle is not None:
            subset = subset[subset['cycle'] == cycle]
            c_label = f"Cycle {int(cycle)}"
        if leadtime is not None:
            subset = subset[subset['leadtime'] == leadtime]
            l_label = f"Leadtime {int(leadtime)}h"

        if len(subset) == 0:
            continue

        actual_n = min(top_n, len(subset))
        title = f"Top {actual_n} — {metric_column} ({c_label}, {l_label})"
        fig = _build_top_table(subset, metric_column, actual_n, title)

        key = _make_combo_key(cycle, leadtime)
        include_js = 'cdn' if first else False
        figure_htmls[key] = fig.to_html(full_html=False, include_plotlyjs=include_js)
        first = False

    # Build cycle options
    cycle_opts = '<option value="All" selected>All Cycles</option>\n'
    for cy in cycles:
        cycle_opts += f'<option value="C{int(cy)}">Cycle {int(cy)}</option>\n'

    # Build leadtime options
    lt_opts = '<option value="All" selected>All Leadtimes</option>\n'
    for lt in leadtimes:
        lt_opts += f'<option value="L{int(lt)}">{int(lt)}h</option>\n'

    # Build divs
    divs_html = ""
    for i, (key, fig_html) in enumerate(figure_htmls.items()):
        display = "block" if i == 0 else "none"
        divs_html += f'<div id="combo_{key}" class="combo-view" style="display:{display};">\n{fig_html}\n</div>\n'

    page_title = f"Top Configurations — {metric_column}"
    html = f"""<!DOCTYPE html>
<html><head>
<title>{page_title}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; }}
    .controls {{ background: #f8f9fa; padding: 12px 20px; border-radius: 8px; margin-bottom: 16px;
                 display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }}
    .controls h2 {{ margin: 0; font-size: 18px; width: 100%; }}
    .filter-group {{ display: flex; align-items: center; gap: 6px; }}
    .filter-group label {{ font-size: 13px; font-weight: 600; color: #555; white-space: nowrap; }}
    .filter-group select {{ font-size: 13px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; }}
</style>
</head><body>
<div class="controls">
    <h2>{page_title}</h2>
    <div class="filter-group">
        <label>Cycle:</label>
        <select id="cycleSelect" onchange="switchView()">{cycle_opts}</select>
    </div>
    <div class="filter-group">
        <label>Leadtime:</label>
        <select id="ltSelect" onchange="switchView()">{lt_opts}</select>
    </div>
</div>
{divs_html}
<script>
function switchView() {{
    var cy = document.getElementById('cycleSelect').value;
    var lt = document.getElementById('ltSelect').value;
    var key = cy + '_' + lt;
    var views = document.querySelectorAll('.combo-view');
    views.forEach(function(v) {{ v.style.display = 'none'; }});
    var el = document.getElementById('combo_' + key);
    if (el) {{
        el.style.display = 'block';
        window.dispatchEvent(new Event('resize'));
    }}
}}
</script>
</body></html>"""

    if output_dir:
        with open(f"{output_dir}/04_top_configs.html", 'w') as f:
            f.write(html)
        print("Saved: 04_top_configs.html")
    return html


def _html_table(data: pd.DataFrame, metric_column: str, top_n: int, title: str) -> str:
    """Build a plain HTML table."""
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


def plot_top10_per_cycle_v12(data: pd.DataFrame, metric_column: str,
                               output_dir: Optional[str] = None) -> Optional[str]:
    """Top 10 per cycle with leadtime filter dropdown."""
    clean = data.dropna(subset=['leadtime']).copy()
    cycles = sorted(clean['cycle'].unique())
    leadtimes = sorted(clean['leadtime'].unique())

    # Build table grids for each leadtime option
    all_grids = {}

    # All leadtimes
    tables_html = _html_table(clean, metric_column, 10, "All Cycles — Top 10")
    for cycle in cycles:
        cycle_data = clean[clean['cycle'] == cycle]
        actual_n = min(10, len(cycle_data))
        tables_html += _html_table(cycle_data, metric_column, actual_n, f"Cycle {int(cycle)} — Top {actual_n}")
    all_grids["All"] = tables_html

    # Per leadtime
    for lt in leadtimes:
        lt_data = clean[clean['leadtime'] == lt]
        tables_html = _html_table(lt_data, metric_column, 10, f"All Cycles — Top 10 ({int(lt)}h)")
        for cycle in cycles:
            cycle_data = lt_data[lt_data['cycle'] == cycle]
            if len(cycle_data) == 0:
                continue
            actual_n = min(10, len(cycle_data))
            tables_html += _html_table(cycle_data, metric_column, actual_n, f"Cycle {int(cycle)} — Top {actual_n} ({int(lt)}h)")
        all_grids[str(int(lt))] = tables_html

    # Build leadtime options
    lt_opts = '<option value="All" selected>All Leadtimes</option>\n'
    for lt in leadtimes:
        lt_opts += f'<option value="{int(lt)}">{int(lt)}h</option>\n'

    # Build grid divs
    grids_html = ""
    for i, (key, tables) in enumerate(all_grids.items()):
        display = "flex" if i == 0 else "none"
        grids_html += f'<div id="grid_{key}" class="grid lt-grid" style="display:{display}; flex-wrap:wrap; gap:16px; justify-content:center;">\n{tables}\n</div>\n'

    html = f"""<!DOCTYPE html>
<html><head>
<title>Top 10 per Cycle — {metric_column}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           margin: 0; padding: 20px; background: #fafafa; }}
    .controls {{ background: #f8f9fa; padding: 12px 20px; border-radius: 8px; margin-bottom: 16px;
                 display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }}
    .controls h2 {{ margin: 0; font-size: 18px; width: 100%; }}
    .filter-group {{ display: flex; align-items: center; gap: 6px; }}
    .filter-group label {{ font-size: 13px; font-weight: 600; color: #555; white-space: nowrap; }}
    .filter-group select {{ font-size: 13px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; }}
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
<div class="controls">
    <h2>Top 10 Configurations per Cycle — {metric_column}</h2>
    <div class="filter-group">
        <label>Leadtime:</label>
        <select id="ltSelect" onchange="switchLt()">{lt_opts}</select>
    </div>
</div>
{grids_html}
<script>
function switchLt() {{
    var lt = document.getElementById('ltSelect').value;
    var grids = document.querySelectorAll('.lt-grid');
    grids.forEach(function(g) {{ g.style.display = 'none'; }});
    var el = document.getElementById('grid_' + lt);
    if (el) el.style.display = 'flex';
}}
</script>
</body></html>"""

    if output_dir:
        with open(f"{output_dir}/04_top10_per_cycle.html", 'w') as f:
            f.write(html)
        print("Saved: 04_top10_per_cycle.html")
    return html
