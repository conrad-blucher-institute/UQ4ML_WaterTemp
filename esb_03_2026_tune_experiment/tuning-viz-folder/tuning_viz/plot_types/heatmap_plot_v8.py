"""
V8 Heatmap: grid rows=param pairs, cols=leadtimes.
Custom HTML. Controls at top. Working auto-resize toggle.
Local color scale. Integer axis labels (no .0).
"""

from typing import Optional
import json
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

HEATMAP_PAIRS = [
    ('activation', 'neurons', 'Activation', 'Hidden Units'),
    ('activation', 'num_layers', 'Activation', 'Num Layers'),
    ('num_layers', 'neurons', 'Num Layers', 'Hidden Units'),
]


def _clean_label(val):
    """Convert numeric values to clean int strings, leave strings alone."""
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, TypeError):
        return str(val)


def _build_heatmap_grid(data: pd.DataFrame, metric_column: str,
                         title_suffix: str = "") -> go.Figure:
    clean = data.dropna(subset=['leadtime']).copy()
    leadtimes = sorted(clean['leadtime'].unique())

    n_rows = len(HEATMAP_PAIRS)
    n_cols = len(leadtimes)

    subplot_titles = []
    for _, _, y_label, x_label in HEATMAP_PAIRS:
        for lt in leadtimes:
            subplot_titles.append(f"{y_label} vs {x_label} — LT {int(lt)}h")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    for row_idx, (y_col, x_col, y_label, x_label) in enumerate(HEATMAP_PAIRS, start=1):
        for col_idx, lt in enumerate(leadtimes, start=1):
            lt_data = clean[clean['leadtime'] == lt]

            pivot = lt_data.pivot_table(
                values=metric_column, index=y_col, columns=x_col, aggfunc='min',
            )

            try:
                pivot = pivot[sorted(pivot.columns, key=lambda x: float(x))]
            except (ValueError, TypeError):
                pass
            try:
                pivot = pivot.reindex(sorted(pivot.index, key=lambda x: float(x)))
            except (ValueError, TypeError):
                pass

            local_min = np.nanmin(pivot.values)
            local_max = np.nanmax(pivot.values)
            show_colorbar = (col_idx == n_cols and row_idx == 1)

            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=[_clean_label(c) for c in pivot.columns],
                    y=[_clean_label(i) for i in pivot.index],
                    colorscale='RdYlGn_r',
                    zmin=local_min, zmax=local_max,
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
                row=row_idx, col=col_idx, type='category',
            )
            fig.update_yaxes(
                title_text=y_label if col_idx == 1 else "",
                row=row_idx, col=col_idx, type='category',
            )

    fig.update_layout(
        title=dict(
            text=(f"Heatmap Grid: Best {metric_column}{title_suffix}"
                  f"<br><sup style='color:#777;'>Each cell = best (min) {metric_column} "
                  f"for that parameter pair. Color scale is local per heatmap.</sup>"),
        ),
        height=280 * n_rows + 120,
        width=280 * n_cols + 120,
    )
    return fig


def plot_heatmap_v8(data: pd.DataFrame, metric_column: str,
                     output_dir: Optional[str] = None) -> Optional[str]:
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()
    cycles = sorted(clean['cycle'].unique())

    views = {}
    fig = _build_heatmap_grid(clean, metric_column, " (All Cycles)")
    views["All Cycles"] = fig

    for cycle in cycles:
        c_int = int(cycle)
        cycle_data = clean[clean['cycle'] == cycle]
        views[f"Cycle {c_int}"] = _build_heatmap_grid(cycle_data, metric_column, f" (Cycle {c_int})")

    cycle_options = ""
    view_divs = ""
    original_sizes = {}

    for i, (label, fig) in enumerate(views.items()):
        safe_id = f"hm_view_{i}"
        selected = "selected" if i == 0 else ""
        display = "block" if i == 0 else "none"
        cycle_options += f'<option value="{safe_id}" {selected}>{label}</option>\n'
        js_flag = 'cdn' if i == 0 else False
        view_divs += f'<div id="{safe_id}" style="display:{display};">\n{fig.to_html(full_html=False, include_plotlyjs=js_flag)}\n</div>\n'
        original_sizes[safe_id] = {'w': fig.layout.width or 1000, 'h': fig.layout.height or 900}

    sizes_json = json.dumps(original_sizes)

    html = f"""<!DOCTYPE html>
<html><head>
<title>Heatmaps — {metric_column}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           margin: 0; padding: 20px; }}
    .controls {{ background: #f8f9fa; padding: 12px 20px; border-radius: 8px; margin-bottom: 16px; }}
    .controls h2 {{ margin: 0 0 8px 0; font-size: 18px; }}
    .filter-bar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 14px; }}
    .filter-group {{ display: flex; align-items: center; gap: 6px; }}
    .filter-group label {{ font-size: 13px; font-weight: 600; color: #555; white-space: nowrap; }}
    .filter-group select {{ font-size: 13px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; }}
    .btn {{ font-size: 12px; padding: 5px 12px; border: 1px solid #ccc; border-radius: 4px;
            background: white; cursor: pointer; }}
    .btn:hover {{ background: #e9ecef; }}
    .btn.active {{ background: #52a8ad; color: white; border-color: #52a8ad; }}
</style>
</head><body>
<div class="controls">
    <h2>Heatmaps — {metric_column}</h2>
    <div class="filter-bar">
        <div class="filter-group">
            <label>Cycle:</label>
            <select id="cycleSelect" onchange="switchCycle()">{cycle_options}</select>
        </div>
        <button id="resizeBtn" class="btn" onclick="toggleResize()">Auto-resize: OFF</button>
    </div>
</div>
{view_divs}
<script>
var originalSizes = {sizes_json};
var autoResize = false;

function switchCycle() {{
    var sel = document.getElementById('cycleSelect');
    document.querySelectorAll('[id^="hm_view_"]').forEach(function(v) {{ v.style.display = 'none'; }});
    document.getElementById(sel.value).style.display = 'block';
    if (autoResize) applyAutoResize();
    else window.dispatchEvent(new Event('resize'));
}}

function toggleResize() {{
    autoResize = !autoResize;
    var btn = document.getElementById('resizeBtn');
    btn.textContent = autoResize ? 'Auto-resize: ON' : 'Auto-resize: OFF';
    btn.classList.toggle('active', autoResize);
    if (autoResize) applyAutoResize(); else applyFixedSize();
}}

function applyAutoResize() {{
    var id = document.getElementById('cycleSelect').value;
    var gd = document.getElementById(id).querySelector('.plotly-graph-div');
    if (gd) Plotly.relayout(gd, {{width: window.innerWidth - 60, height: window.innerHeight - 120}});
}}

function applyFixedSize() {{
    var id = document.getElementById('cycleSelect').value;
    var gd = document.getElementById(id).querySelector('.plotly-graph-div');
    if (gd && originalSizes[id]) Plotly.relayout(gd, {{width: originalSizes[id].w, height: originalSizes[id].h}});
}}

window.addEventListener('resize', function() {{ if (autoResize) applyAutoResize(); }});
</script>
</body></html>"""

    if output_dir:
        with open(f"{output_dir}/02_heatmaps.html", 'w') as f:
            f.write(html)
        print("Saved: 02_heatmaps.html")
    return html
