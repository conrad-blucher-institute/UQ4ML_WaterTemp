"""
V7 Scatter: x-axis selector, all filter dropdowns, color-by option.
No legend. Shows "Models (N)" count. Color #52a8ad default, or color by param.
"""

from typing import Optional
import json as _json
import pandas as pd

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_scatter_v7(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[str]:
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()

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

    def _sorted_unique(col, as_int=True):
        vals = clean[col].unique()
        return sorted([int(v) for v in vals]) if as_int else sorted([str(v) for v in vals])

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

    # Build unique values map for color-by
    unique_vals_json = _json.dumps({
        'activation': activations,
        'num_layers': num_layers_vals,
        'neurons': neurons_vals,
        'leadtime': leadtime_vals,
        'cycle': cycle_vals,
    })

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
    .separator {{ border-left: 2px solid #ddd; height: 24px; }}
    .model-count {{ font-size: 13px; font-weight: 600; color: #52a8ad; padding: 4px 10px;
                    background: rgba(82, 168, 173, 0.1); border-radius: 4px; }}
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
        <div class="filter-group">
            <label>Color by:</label>
            <select id="colorBy" onchange="rebuild()">
                <option value="none" selected>None</option>
                <option value="activation">Activation</option>
                <option value="num_layers">Num Layers</option>
                <option value="neurons">Hidden Units</option>
                <option value="leadtime">Leadtime</option>
                <option value="cycle">Cycle</option>
            </select>
        </div>
        <div class="separator"></div>
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
        <span id="modelCount" class="model-count">Models (0)</span>
    </div>
</div>
<div id="plotDiv"></div>

<script>
var allData = {data_json};
var metricCol = "{metric_column}";
var uniqueVals = {unique_vals_json};

var xAxisLabels = {{
    'neurons': 'Hidden Units',
    'num_layers': 'Number of Layers',
    'leadtime': 'Leadtime (h)',
    'cycle': 'Cycle',
    'activation': 'Activation Function'
}};

var colorPalette = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
    '#a65628','#f781bf','#66c2a5','#fc8d62','#8da0cb',
    '#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'
];

function rebuild() {{
    var xParam = document.getElementById('xAxis').value;
    var colorBy = document.getElementById('colorBy').value;
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

    document.getElementById('modelCount').textContent = 'Models (' + filtered.length + ')';

    var traces = [];

    if (colorBy === 'none' || colorBy === xParam) {{
        var xVals = filtered.map(function(d) {{ return String(d[xParam]); }});
        var yVals = filtered.map(function(d) {{ return d.metric; }});
        var hoverText = filtered.map(function(d) {{
            return 'Act: ' + d.activation + '<br>Layers: ' + d.num_layers +
                   '<br>Neurons: ' + d.neurons + '<br>LT: ' + d.leadtime + 'h' +
                   '<br>Cycle: ' + d.cycle + '<br>' + metricCol + ': ' + d.metric.toFixed(4);
        }});

        traces.push({{
            x: xVals, y: yVals, mode: 'markers', type: 'scatter',
            marker: {{ size: 6, color: 'rgba(82, 168, 173, 0.55)',
                       line: {{ width: 0.5, color: 'rgba(60, 130, 135, 0.4)' }} }},
            text: hoverText, hoverinfo: 'text', showlegend: false
        }});
    }} else {{
        var groups = {{}};
        filtered.forEach(function(d) {{
            var key = String(d[colorBy]);
            if (!groups[key]) groups[key] = [];
            groups[key].push(d);
        }});

        var keys = Object.keys(groups);
        if (colorBy !== 'activation') {{
            keys.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
        }} else {{
            keys.sort();
        }}

        keys.forEach(function(key, ci) {{
            var sub = groups[key];
            traces.push({{
                x: sub.map(function(d) {{ return String(d[xParam]); }}),
                y: sub.map(function(d) {{ return d.metric; }}),
                mode: 'markers', type: 'scatter',
                name: key, showlegend: true,
                marker: {{ size: 6, color: colorPalette[ci % colorPalette.length], opacity: 0.6,
                           line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }} }},
                text: sub.map(function(d) {{
                    return 'Act: ' + d.activation + '<br>Layers: ' + d.num_layers +
                           '<br>Neurons: ' + d.neurons + '<br>LT: ' + d.leadtime + 'h' +
                           '<br>Cycle: ' + d.cycle + '<br>' + metricCol + ': ' + d.metric.toFixed(4);
                }}),
                hoverinfo: 'text'
            }});
        }});
    }}

    var allX = [];
    traces.forEach(function(t) {{ allX = allX.concat(t.x); }});
    var uniqueX = [...new Set(allX)];
    if (xParam !== 'activation') {{
        uniqueX.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
    }} else {{ uniqueX.sort(); }}

    var showLeg = (colorBy !== 'none' && colorBy !== xParam);

    var layout = {{
        xaxis: {{ title: xAxisLabels[xParam], type: 'category',
                  categoryorder: 'array', categoryarray: uniqueX }},
        yaxis: {{ title: metricCol }},
        height: 550, margin: {{ t: 30, b: 60, l: 70, r: 30 }},
        hovermode: 'closest', showlegend: showLeg,
        legend: {{ title: {{ text: xAxisLabels[colorBy] || '' }} }}
    }};

    Plotly.newPlot('plotDiv', traces, layout, {{ responsive: true }});
}}

rebuild();
</script>
</body></html>"""

    if output_dir:
        with open(f"{output_dir}/01_scatter.html", 'w') as f:
            f.write(html)
        print("Saved: 01_scatter.html")
    return html
