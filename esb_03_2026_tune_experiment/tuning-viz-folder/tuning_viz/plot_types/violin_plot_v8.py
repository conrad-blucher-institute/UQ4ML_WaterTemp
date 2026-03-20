"""
V8 Violin: same controls as boxplot. Points offset to the right.
Filter dropdowns, Models(N), y-axis starts at 0, auto-height.
"""

from typing import Optional
import json as _json
import pandas as pd

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_violin_v8(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[str]:
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
        'activation': 'Activation Function', 'num_layers': 'Number of Layers',
        'neurons': 'Hidden Units', 'leadtime': 'Leadtime (h)', 'cycle': 'Cycle',
    }
    param_labels_json = _json.dumps(param_labels)

    def _sorted_unique(col, as_int=True):
        vals = clean[col].unique()
        return sorted([int(v) for v in vals]) if as_int else sorted([str(v) for v in vals])

    def _make_options(values, label="All"):
        opts = f'<option value="ALL" selected>{label}</option>\n'
        for v in values:
            opts += f'<option value="{v}">{v}</option>\n'
        return opts

    color_options = '<option value="none" selected>None</option>\n'
    for p in param_labels:
        color_options += f'<option value="{p}">{param_labels[p]}</option>\n'

    html = f"""<!DOCTYPE html>
<html><head>
<title>Violin: {metric_column}</title>
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
    .btn {{ font-size: 12px; padding: 5px 12px; border: 1px solid #ccc; border-radius: 4px;
            background: white; cursor: pointer; }}
    .btn:hover {{ background: #e9ecef; }}
    .btn.active {{ background: #52a8ad; color: white; border-color: #52a8ad; }}
    .model-count {{ font-size: 13px; font-weight: 600; color: #52a8ad; padding: 4px 10px;
                    background: rgba(82, 168, 173, 0.1); border-radius: 4px; }}
    #plotDiv {{ width: 100%; }}
</style>
</head><body>
<div class="controls">
    <h2>Violin Plot: {metric_column}</h2>
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
        <button id="pointsBtn" class="btn" onclick="togglePoints()">Points: OFF</button>
        <button id="legendBtn" class="btn" onclick="toggleLegend()">Legend: OFF</button>
        <div class="separator"></div>
        <div class="filter-group">
            <label>Activation:</label>
            <select id="fAct" onchange="rebuild()">{_make_options(_sorted_unique('activation', False))}</select>
        </div>
        <div class="filter-group">
            <label>Layers:</label>
            <select id="fLayers" onchange="rebuild()">{_make_options(_sorted_unique('num_layers'))}</select>
        </div>
        <div class="filter-group">
            <label>Neurons:</label>
            <select id="fNeurons" onchange="rebuild()">{_make_options(_sorted_unique('neurons'))}</select>
        </div>
        <div class="filter-group">
            <label>Leadtime:</label>
            <select id="fLT" onchange="rebuild()">{_make_options(_sorted_unique('leadtime'))}</select>
        </div>
        <div class="filter-group">
            <label>Cycle:</label>
            <select id="fCy" onchange="rebuild()">{_make_options(_sorted_unique('cycle'))}</select>
        </div>
        <span id="modelCount" class="model-count">Models (0)</span>
    </div>
</div>
<div id="plotDiv"></div>

<script>
var allData = {data_json};
var metricCol = "{metric_column}";
var paramLabels = {param_labels_json};
var showPoints = false;
var showLegend = false;

var colors = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
    '#a65628','#f781bf','#999999','#66c2a5','#fc8d62',
    '#8da0cb','#e78ac3','#a6d854','#ffd92f','#b3b3b3'
];

function applyFilter() {{
    var fAct = document.getElementById('fAct').value;
    var fLayers = document.getElementById('fLayers').value;
    var fNeurons = document.getElementById('fNeurons').value;
    var fLT = document.getElementById('fLT').value;
    var fCy = document.getElementById('fCy').value;

    return allData.filter(function(d) {{
        if (fAct !== 'ALL' && d.activation !== fAct) return false;
        if (fLayers !== 'ALL' && d.num_layers !== parseInt(fLayers)) return false;
        if (fNeurons !== 'ALL' && d.neurons !== parseInt(fNeurons)) return false;
        if (fLT !== 'ALL' && d.leadtime !== parseInt(fLT)) return false;
        if (fCy !== 'ALL' && d.cycle !== parseInt(fCy)) return false;
        return true;
    }});
}}

function togglePoints() {{
    showPoints = !showPoints;
    var btn = document.getElementById('pointsBtn');
    btn.textContent = showPoints ? 'Points: ON' : 'Points: OFF';
    btn.classList.toggle('active', showPoints);
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
    var filtered = applyFilter();

    document.getElementById('modelCount').textContent = 'Models (' + filtered.length + ')';

    var traces = [];
    var pointsMode = showPoints ? 'all' : false;

    if (colorBy === 'none' || colorBy === xParam) {{
        var groups = {{}};
        filtered.forEach(function(d) {{
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
                y: groups[key], type: 'violin', name: key,
                box: {{ visible: true }}, meanline: {{ visible: true }},
                line: {{ color: colors[i % colors.length] }},
                points: pointsMode, jitter: 0.5, pointpos: 0.8,
                marker: {{ size: 3, opacity: 0.5 }},
                side: 'positive'
            }});
        }});
    }} else {{
        var colorVals = [...new Set(filtered.map(function(d) {{ return String(d[colorBy]); }}))];
        if (colorBy !== 'activation') {{
            colorVals.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
        }} else {{ colorVals.sort(); }}

        colorVals.forEach(function(cv, ci) {{
            var sub = filtered.filter(function(d) {{ return String(d[colorBy]) === cv; }});
            traces.push({{
                x: sub.map(function(d) {{ return String(d[xParam]); }}),
                y: sub.map(function(d) {{ return d.metric; }}),
                type: 'violin', name: cv,
                box: {{ visible: true }}, meanline: {{ visible: true }},
                line: {{ color: colors[ci % colors.length] }},
                points: pointsMode, jitter: 0.5, pointpos: 0.8,
                marker: {{ size: 3, opacity: 0.5 }}
            }});
        }});
    }}

    var h = Math.max(500, window.innerHeight - 140);

    var layout = {{
        xaxis: {{ title: paramLabels[xParam], type: 'category' }},
        yaxis: {{ title: metricCol, rangemode: 'tozero' }},
        height: h,
        margin: {{ t: 30, b: 60, l: 70, r: 30 }},
        violinmode: 'group',
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
        with open(f"{output_dir}/03b_violin.html", 'w') as f:
            f.write(html)
        print("Saved: 03b_violin.html")
    return html
