"""
V6 Scatter: x-axis selector dropdown, filter dropdowns for all hyperparameters.
Single dark color for all dots. Filtered traces fully hidden (not just legendonly).
JS rebuilds plot from data on any filter/axis change.
"""

from typing import Optional
import json as _json
import pandas as pd

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_scatter_v6(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[str]:
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()

    # Prepare data as JSON-serializable records
    records = []
    for _, row in clean.iterrows():
        records.append({
            'activation': str(row['activation']),
            'num_layers': int(row['num_layers']),
            'neurons': int(row['neurons']),
            'leadtime': int(row['leadtime']),
            'cycle': int(row['cycle']),
            'metric': float(row[metric_column]),
        })

    data_json = _json.dumps(records)

    # Get unique values for each param (for filter dropdowns)
    def _sorted_unique(col, as_int=True):
        vals = clean[col].unique()
        if as_int:
            return sorted([int(v) for v in vals])
        return sorted([str(v) for v in vals])

    activations = _sorted_unique('activation', as_int=False)
    num_layers_vals = _sorted_unique('num_layers')
    neurons_vals = _sorted_unique('neurons')
    leadtime_vals = _sorted_unique('leadtime')
    cycle_vals = _sorted_unique('cycle')

    def _make_options(values, label="All"):
        opts = f'<option value="ALL" selected>{label}</option>\n'
        for v in values:
            opts += f'<option value="{v}">{v}</option>\n'
        return opts

    html = f"""<!DOCTYPE html>
<html><head>
<title>Scatter: {metric_column}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           margin: 0; padding: 20px; }}
    .controls {{ background: #f8f9fa; padding: 12px 20px; border-radius: 8px; margin-bottom: 8px; }}
    .controls h2 {{ margin: 0 0 8px 0; font-size: 18px; }}
    .filter-bar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 14px; }}
    .filter-group {{ display: flex; align-items: center; gap: 6px; }}
    .filter-group label {{ font-size: 13px; font-weight: 600; color: #555; white-space: nowrap; }}
    .filter-group select {{ font-size: 13px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; }}
    .btn {{ font-size: 12px; padding: 5px 12px; border: 1px solid #ccc; border-radius: 4px;
            background: white; cursor: pointer; }}
    .btn:hover {{ background: #e9ecef; }}
    #plotDiv {{ width: 100%; }}
</style>
</head><body>
<div class="controls">
    <h2>Hyperparameter Scatter: {metric_column}</h2>
    <div class="filter-bar">
        <div class="filter-group">
            <label>X Axis:</label>
            <select id="xAxis" onchange="rebuild()">
                <option value="neurons" selected>Hidden Units</option>
                <option value="num_layers">Num Layers</option>
                <option value="leadtime">Leadtime (h)</option>
                <option value="cycle">Cycle</option>
                <option value="activation">Activation</option>
            </select>
        </div>
        <div style="border-left: 2px solid #ddd; height: 24px;"></div>
        <div class="filter-group">
            <label>Activation:</label>
            <select id="fAct" onchange="rebuild()">{_make_options(activations)}</select>
        </div>
        <div class="filter-group">
            <label>Layers:</label>
            <select id="fLayers" onchange="rebuild()">{_make_options(num_layers_vals)}</select>
        </div>
        <div class="filter-group">
            <label>Neurons:</label>
            <select id="fNeurons" onchange="rebuild()">{_make_options(neurons_vals)}</select>
        </div>
        <div class="filter-group">
            <label>Leadtime:</label>
            <select id="fLT" onchange="rebuild()">{_make_options(leadtime_vals)}</select>
        </div>
        <div class="filter-group">
            <label>Cycle:</label>
            <select id="fCy" onchange="rebuild()">{_make_options(cycle_vals)}</select>
        </div>
        <button class="btn" onclick="toggleLegend()">Toggle Legend</button>
    </div>
</div>
<div id="plotDiv"></div>

<script>
var allData = {data_json};
var metricCol = "{metric_column}";
var showLegend = false;

var xAxisLabels = {{
    'neurons': 'Hidden Units',
    'num_layers': 'Number of Layers',
    'leadtime': 'Leadtime (h)',
    'cycle': 'Cycle',
    'activation': 'Activation Function'
}};

function rebuild() {{
    var xParam = document.getElementById('xAxis').value;
    var fAct = document.getElementById('fAct').value;
    var fLayers = document.getElementById('fLayers').value;
    var fNeurons = document.getElementById('fNeurons').value;
    var fLT = document.getElementById('fLT').value;
    var fCy = document.getElementById('fCy').value;

    var filtered = allData.filter(function(d) {{
        if (fAct !== 'ALL' && d.activation !== fAct) return false;
        if (fLayers !== 'ALL' && d.num_layers !== parseInt(fLayers)) return false;
        if (fNeurons !== 'ALL' && d.neurons !== parseInt(fNeurons)) return false;
        if (fLT !== 'ALL' && d.leadtime !== parseInt(fLT)) return false;
        if (fCy !== 'ALL' && d.cycle !== parseInt(fCy)) return false;
        return true;
    }});

    var xVals = filtered.map(function(d) {{ return String(d[xParam]); }});
    var yVals = filtered.map(function(d) {{ return d.metric; }});
    var hoverText = filtered.map(function(d) {{
        return 'Act: ' + d.activation + '<br>Layers: ' + d.num_layers +
               '<br>Neurons: ' + d.neurons + '<br>LT: ' + d.leadtime + 'h' +
               '<br>Cycle: ' + d.cycle + '<br>' + metricCol + ': ' + d.metric.toFixed(4);
    }});

    var trace = {{
        x: xVals,
        y: yVals,
        mode: 'markers',
        type: 'scatter',
        marker: {{
            size: 6,
            color: 'rgba(30, 30, 30, 0.45)',
            line: {{ width: 0.5, color: 'rgba(0,0,0,0.2)' }}
        }},
        text: hoverText,
        hoverinfo: 'text',
        name: 'configs (' + filtered.length + ')',
        showlegend: showLegend
    }};

    // Sort x categories
    var uniqueX = [...new Set(xVals)];
    if (xParam !== 'activation') {{
        uniqueX.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
    }} else {{
        uniqueX.sort();
    }}

    var layout = {{
        xaxis: {{
            title: xAxisLabels[xParam],
            type: 'category',
            categoryorder: 'array',
            categoryarray: uniqueX
        }},
        yaxis: {{ title: metricCol }},
        height: 550,
        margin: {{ t: 30, b: 60, l: 70, r: 30 }},
        hovermode: 'closest',
        showlegend: showLegend
    }};

    Plotly.newPlot('plotDiv', [trace], layout, {{ responsive: true }});
}}

function toggleLegend() {{
    showLegend = !showLegend;
    rebuild();
}}

rebuild();
</script>
</body></html>"""

    if output_dir:
        path = f"{output_dir}/01_scatter.html"
        with open(path, 'w') as f:
            f.write(html)
        print("Saved: 01_scatter.html")

    return html
