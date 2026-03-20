"""
V5 Scatter plots: combined JS-based filtering for activation, leadtime, cycle.
Title on top, filter bar below, dropdowns to the right of labels.
Legend toggle button. Cycle in legend names. Single output file.
"""

from typing import Optional, List
import json as _json
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_scatter_v5(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Single scatter HTML with combined dropdown filters for activation, leadtime, cycle.
    Filters combine (AND logic). Legend toggle button.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()
    clean['neurons_cat'] = clean['neurons'].astype(int).astype(str)
    clean['num_layers_cat'] = clean['num_layers'].astype(int).astype(str)
    clean['leadtime_label'] = clean['leadtime'].astype(int).astype(str) + 'h'
    clean['cycle_label'] = 'C' + clean['cycle'].astype(int).astype(str)

    x_axes = [
        ('neurons_cat', 'Hidden Units'),
        ('num_layers_cat', 'Number of Layers'),
    ]

    activations = sorted(clean['activation'].unique())
    leadtime_labels = sorted(clean['leadtime_label'].unique(), key=lambda x: int(x.replace('h', '')))
    cycle_labels = sorted(clean['cycle_label'].unique(), key=lambda x: int(x.replace('C', '')))

    act_colors = px.colors.qualitative.Set1
    act_color_map = {act: act_colors[i % len(act_colors)] for i, act in enumerate(activations)}

    symbol_list = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'star', 'hexagon']
    lt_symbol_map = {lt: symbol_list[i % len(symbol_list)] for i, lt in enumerate(leadtime_labels)}

    fig = make_subplots(
        rows=len(x_axes), cols=1,
        subplot_titles=[label for _, label in x_axes],
        vertical_spacing=0.12,
    )

    trace_meta = []  # {activation, leadtime, cycle} per trace
    seen_legend = set()

    for row_idx, (x_col, x_label) in enumerate(x_axes, start=1):
        cat_order = sorted(clean[x_col].unique(), key=lambda x: int(x))

        for activation in activations:
            for lt_label in leadtime_labels:
                for cy_label in cycle_labels:
                    subset = clean[
                        (clean['activation'] == activation) &
                        (clean['leadtime_label'] == lt_label) &
                        (clean['cycle_label'] == cy_label)
                    ]
                    if subset.empty:
                        continue

                    legend_name = f"{activation} | {lt_label} | {cy_label}"
                    show = legend_name not in seen_legend
                    seen_legend.add(legend_name)

                    trace_meta.append({
                        'activation': activation,
                        'leadtime': lt_label,
                        'cycle': cy_label,
                    })

                    fig.add_trace(
                        go.Scatter(
                            x=subset[x_col],
                            y=subset[metric_column],
                            mode='markers',
                            name=legend_name,
                            legendgroup=legend_name,
                            showlegend=show,
                            marker=dict(
                                size=7,
                                color=act_color_map[activation],
                                symbol=lt_symbol_map[lt_label],
                                opacity=0.7,
                                line=dict(width=0.5, color='white'),
                            ),
                            text=[
                                f"Activation: {a}<br>Neurons: {n}<br>Layers: {l}<br>"
                                f"Leadtime: {lt}<br>Cycle: {cy}<br>{metric_column}: {m:.4f}"
                                for a, n, l, lt, cy, m in zip(
                                    subset['activation'], subset['neurons'],
                                    subset['num_layers'], subset['leadtime_label'],
                                    subset['cycle_label'], subset[metric_column],
                                )
                            ],
                            hoverinfo='text',
                        ),
                        row=row_idx, col=1,
                    )

        fig.update_xaxes(
            title_text=x_label, row=row_idx, col=1,
            type='category', categoryorder='array', categoryarray=cat_order,
        )
        fig.update_yaxes(title_text=metric_column, row=row_idx, col=1)

    fig.update_layout(
        height=450 * len(x_axes),
        hovermode='closest',
        legend=dict(title="Activation | Leadtime | Cycle", itemsizing='constant'),
        margin=dict(t=30),
    )

    # Build custom HTML with JS combined filtering
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    meta_json = _json.dumps(trace_meta)

    def _make_options(values):
        opts = '<option value="ALL" selected>All</option>\n'
        for v in values:
            opts += f'<option value="{v}">{v}</option>\n'
        return opts

    html = f"""<!DOCTYPE html>
<html><head>
<title>Scatter: {metric_column}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; }}
    .controls {{ background: #f8f9fa; padding: 12px 20px; border-radius: 8px; margin-bottom: 8px; }}
    .controls h2 {{ margin: 0 0 8px 0; font-size: 18px; }}
    .filter-bar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 14px; }}
    .filter-group {{ display: flex; align-items: center; gap: 6px; }}
    .filter-group label {{ font-size: 13px; font-weight: 600; color: #555; white-space: nowrap; }}
    .filter-group select {{ font-size: 13px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; }}
    .btn {{ font-size: 12px; padding: 5px 12px; border: 1px solid #ccc; border-radius: 4px;
            background: white; cursor: pointer; }}
    .btn:hover {{ background: #e9ecef; }}
</style>
</head><body>
<div class="controls">
    <h2>Hyperparameter Scatter: {metric_column}</h2>
    <div class="filter-bar">
        <div class="filter-group">
            <label>Activation:</label>
            <select id="filterAct" onchange="applyFilters()">{_make_options(activations)}</select>
        </div>
        <div class="filter-group">
            <label>Leadtime:</label>
            <select id="filterLT" onchange="applyFilters()">{_make_options(leadtime_labels)}</select>
        </div>
        <div class="filter-group">
            <label>Cycle:</label>
            <select id="filterCy" onchange="applyFilters()">{_make_options(cycle_labels)}</select>
        </div>
        <button class="btn" onclick="toggleLegend()">Toggle Legend</button>
    </div>
</div>
<div id="plotArea">{fig_html}</div>
<script>
var traceMeta = {meta_json};
var legendVisible = true;

function applyFilters() {{
    var act = document.getElementById('filterAct').value;
    var lt = document.getElementById('filterLT').value;
    var cy = document.getElementById('filterCy').value;

    var vis = traceMeta.map(function(m) {{
        var matchAct = (act === 'ALL' || m.activation === act);
        var matchLT = (lt === 'ALL' || m.leadtime === lt);
        var matchCy = (cy === 'ALL' || m.cycle === cy);
        return (matchAct && matchLT && matchCy) ? true : 'legendonly';
    }});

    var gd = document.getElementById('plotArea').querySelector('.plotly-graph-div');
    Plotly.restyle(gd, {{'visible': vis}});
}}

function toggleLegend() {{
    legendVisible = !legendVisible;
    var gd = document.getElementById('plotArea').querySelector('.plotly-graph-div');
    Plotly.relayout(gd, {{'showlegend': legendVisible}});
}}
</script>
</body></html>"""

    if output_dir:
        path = f"{output_dir}/01_scatter.html"
        with open(path, 'w') as f:
            f.write(html)
        print("Saved: 01_scatter.html")

    return html
