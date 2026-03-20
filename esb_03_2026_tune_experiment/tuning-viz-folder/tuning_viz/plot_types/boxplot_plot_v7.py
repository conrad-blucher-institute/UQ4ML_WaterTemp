"""
V7 Boxplot: x-axis selector + color-by. No cycle dropdown (cycle is an x option).
Legend toggle. Auto-height. Jitter strip overlay toggle.
"""

from typing import Optional
import json as _json
import pandas as pd

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_boxplot_v7(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[str]:
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

    param_labels = {
        'activation': 'Activation Function',
        'num_layers': 'Number of Layers',
        'neurons': 'Hidden Units',
        'leadtime': 'Leadtime (h)',
        'cycle': 'Cycle',
    }
    param_labels_json = _json.dumps(param_labels)

    params = list(param_labels.keys())
    color_options = '<option value="none" selected>None</option>\n'
    for p in params:
        color_options += f'<option value="{p}">{param_labels[p]}</option>\n'

    html = f"""<!DOCTYPE html>
<html><head>
<title>Boxplot: {metric_column}</title>
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
    .btn.active {{ background: #52a8ad; color: white; border-color: #52a8ad; }}
    #plotDiv {{ width: 100%; }}
</style>
</head><body>
<div class="controls">
    <h2>Boxplot: {metric_column}</h2>
    <div class="filter-bar">
        <div class="filter-group">
            <label>X Axis:</label>
            <select id="xAxis" onchange="rebuild()">
                <option value="activation" selected>Activation Function</option>
                <option value="num_layers">Number of Layers</option>
                <option value="neurons">Hidden Units</option>
                <option value="leadtime">Leadtime (h)</option>
                <option value="cycle">Cycle</option>
            </select>
        </div>
        <div class="filter-group">
            <label>Color by:</label>
            <select id="colorBy" onchange="rebuild()">
                {color_options}
            </select>
        </div>
        <button id="jitterBtn" class="btn" onclick="toggleJitter()">Strip points: OFF</button>
        <button id="legendBtn" class="btn" onclick="toggleLegend()">Legend: OFF</button>
    </div>
</div>
<div id="plotDiv"></div>

<script>
var allData = {data_json};
var metricCol = "{metric_column}";
var paramLabels = {param_labels_json};
var showJitter = false;
var showLegend = false;

var colors = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
    '#a65628','#f781bf','#999999','#66c2a5','#fc8d62',
    '#8da0cb','#e78ac3','#a6d854','#ffd92f','#b3b3b3'
];

function toggleJitter() {{
    showJitter = !showJitter;
    var btn = document.getElementById('jitterBtn');
    btn.textContent = showJitter ? 'Strip points: ON' : 'Strip points: OFF';
    btn.classList.toggle('active', showJitter);
    rebuild();
}}

function toggleLegend() {{
    showLegend = !showLegend;
    var btn = document.getElementById('legendBtn');
    btn.textContent = showLegend ? 'Legend: ON' : 'Legend: OFF';
    btn.classList.toggle('active', showLegend);
    var gd = document.getElementById('plotDiv');
    Plotly.relayout(gd, {{'showlegend': showLegend}});
}}

function rebuild() {{
    var xParam = document.getElementById('xAxis').value;
    var colorBy = document.getElementById('colorBy').value;

    var traces = [];

    if (colorBy === 'none' || colorBy === xParam) {{
        var groups = {{}};
        allData.forEach(function(d) {{
            var key = String(d[xParam]);
            if (!groups[key]) groups[key] = [];
            groups[key].push(d.metric);
        }});

        var sortedKeys = Object.keys(groups);
        if (xParam !== 'activation') {{
            sortedKeys.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
        }} else {{ sortedKeys.sort(); }}

        sortedKeys.forEach(function(key, i) {{
            traces.push({{
                y: groups[key], type: 'box', name: key, boxmean: 'sd',
                marker: {{ color: colors[i % colors.length] }},
                boxpoints: showJitter ? 'all' : false,
                jitter: 0.4, pointpos: 0,
                marker: {{ color: colors[i % colors.length], size: 3, opacity: 0.4 }},
            }});
        }});
    }} else {{
        var colorVals = [...new Set(allData.map(function(d) {{ return String(d[colorBy]); }}))];
        if (colorBy !== 'activation') {{
            colorVals.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
        }} else {{ colorVals.sort(); }}

        colorVals.forEach(function(cv, ci) {{
            var sub = allData.filter(function(d) {{ return String(d[colorBy]) === cv; }});
            traces.push({{
                x: sub.map(function(d) {{ return String(d[xParam]); }}),
                y: sub.map(function(d) {{ return d.metric; }}),
                type: 'box', name: cv, boxmean: 'sd',
                marker: {{ color: colors[ci % colors.length], size: 3, opacity: 0.4 }},
                boxpoints: showJitter ? 'all' : false,
                jitter: 0.4, pointpos: 0,
            }});
        }});
    }}

    var h = Math.max(500, window.innerHeight - 140);

    var layout = {{
        xaxis: {{ title: paramLabels[xParam], type: 'category' }},
        yaxis: {{ title: metricCol }},
        height: h,
        margin: {{ t: 30, b: 60, l: 70, r: 30 }},
        boxmode: 'group',
        hovermode: 'closest',
        showlegend: showLegend,
        legend: {{ title: {{ text: paramLabels[colorBy] || '' }} }}
    }};

    Plotly.newPlot('plotDiv', traces, layout, {{ responsive: true }});
}}

window.addEventListener('resize', function() {{ rebuild(); }});
rebuild();
</script>
</body></html>"""

    if output_dir:
        with open(f"{output_dir}/03_boxplot.html", 'w') as f:
            f.write(html)
        print("Saved: 03_boxplot.html")
    return html
