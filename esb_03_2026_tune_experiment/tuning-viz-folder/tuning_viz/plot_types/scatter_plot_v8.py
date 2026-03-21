"""
V8 Scatter: x-axis selector, y-axis metric selector, filter dropdowns,
color-by with side-by-side grouping. Models(N) counter. #52a8ad default color.
"""

from typing import Optional
import json as _json
import pandas as pd

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_scatter_v8(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[str]:
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()

    # Detect all numeric metric columns available in the data
    _skip = {'run_id', 'iteration', 'leadtime', 'cycle', 'num_layers', 'neurons'}
    metric_cols = [c for c in clean.columns
                   if c not in _skip
                   and pd.api.types.is_numeric_dtype(clean[c])
                   and clean[c].notna().any()
                   and c not in ('metric_type',)]
    # Ensure the requested metric_column is first in the list
    if metric_column in metric_cols:
        metric_cols.remove(metric_column)
    metric_cols.insert(0, metric_column)

    # Build records with ALL metric columns so the JS can switch y-axis
    records = []
    for _, row in clean.iterrows():
        rec = {
            'activation': str(row['activation']),
            'num_layers': int(row['num_layers']),
            'neurons': int(row['neurons']),
            'leadtime': int(row['leadtime']),
            'cycle': int(row['cycle']),
        }
        for mc in metric_cols:
            try:
                rec[mc] = float(row[mc])
            except (ValueError, TypeError):
                rec[mc] = None
        records.append(rec)

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

    # Build y-axis metric dropdown options (first one = selected)
    y_axis_options = ''
    for i, mc in enumerate(metric_cols):
        sel = ' selected' if i == 0 else ''
        y_axis_options += f'<option value="{mc}"{sel}>{mc}</option>\n'

    metric_cols_json = _json.dumps(metric_cols)

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
    <h2>Hyperparameter Scatter</h2>
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
            <label>Y Axis:</label>
            <select id="yAxis" onchange="rebuild()">
                {y_axis_options}
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
        <span id="missingCount" class="model-count" style="color: #c0392b; background: rgba(192, 57, 43, 0.1);">Missing (0)</span>
    </div>
</div>
<div id="plotDiv"></div>

<script>
var allData = {data_json};
var metricCols = {metric_cols_json};

var xAxisLabels = {{
    'neurons': 'Hidden Units', 'num_layers': 'Number of Layers',
    'leadtime': 'Leadtime (h)', 'cycle': 'Cycle', 'activation': 'Activation Function'
}};

var colorPalette = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
    '#a65628','#f781bf','#66c2a5','#fc8d62','#8da0cb',
    '#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'
];

function makeHover(d, yMetric) {{
    return 'Act: ' + d.activation + '<br>Layers: ' + d.num_layers +
           '<br>Neurons: ' + d.neurons + '<br>LT: ' + d.leadtime + 'h' +
           '<br>Cycle: ' + d.cycle + '<br>' + yMetric + ': ' +
           (d[yMetric] != null ? d[yMetric].toFixed(4) : 'N/A');
}}

function rebuild() {{
    var xParam = document.getElementById('xAxis').value;
    var yMetric = document.getElementById('yAxis').value;
    var colorBy = document.getElementById('colorBy').value;
    var fAct = document.getElementById('fAct').value;
    var fLayers = document.getElementById('fLayers').value;
    var fNeurons = document.getElementById('fNeurons').value;
    var fLT = document.getElementById('fLT').value;
    var fCy = document.getElementById('fCy').value;

    // First filter by hyperparameter selections only
    var paramFiltered = allData.filter(function(d) {{
        if (fAct !== 'ALL' && d.activation !== fAct) return false;
        if (fLayers !== 'ALL' && d.num_layers !== parseInt(fLayers)) return false;
        if (fNeurons !== 'ALL' && d.neurons !== parseInt(fNeurons)) return false;
        if (fLT !== 'ALL' && d.leadtime !== parseInt(fLT)) return false;
        if (fCy !== 'ALL' && d.cycle !== parseInt(fCy)) return false;
        return true;
    }});

    // Then separate: rows with valid y metric vs rows missing it
    var filtered = paramFiltered.filter(function(d) {{ return d[yMetric] != null; }});
    var missingCount = paramFiltered.length - filtered.length;

    document.getElementById('modelCount').textContent = 'Models (' + filtered.length + ')';
    var missingEl = document.getElementById('missingCount');
    if (missingCount > 0) {{
        missingEl.textContent = 'Missing ' + yMetric + ' (' + missingCount + ')';
        missingEl.style.display = '';
    }} else {{
        missingEl.style.display = 'none';
    }}

    var traces = [];
    var useColor = (colorBy !== 'none' && colorBy !== xParam);

    if (!useColor) {{
        traces.push({{
            x: filtered.map(function(d) {{ return String(d[xParam]); }}),
            y: filtered.map(function(d) {{ return d[yMetric]; }}),
            mode: 'markers', type: 'scatter',
            marker: {{ size: 6, color: 'rgba(82, 168, 173, 0.55)',
                       line: {{ width: 0.5, color: 'rgba(60, 130, 135, 0.4)' }} }},
            text: filtered.map(function(d) {{ return makeHover(d, yMetric); }}),
            hoverinfo: 'text', showlegend: false
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
        }} else {{ keys.sort(); }}

        var nGroups = keys.length;
        var spread = 0.6;
        var step = nGroups > 1 ? spread / (nGroups - 1) : 0;
        var startOffset = nGroups > 1 ? -spread / 2 : 0;

        keys.forEach(function(key, ci) {{
            var sub = groups[key];
            var offset = startOffset + ci * step;

            var allXRaw = filtered.map(function(d) {{ return String(d[xParam]); }});
            var uniqueX = [...new Set(allXRaw)];
            if (xParam !== 'activation') {{
                uniqueX.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
            }} else {{ uniqueX.sort(); }}
            var xMap = {{}};
            uniqueX.forEach(function(v, i) {{ xMap[v] = i; }});

            var xNums = sub.map(function(d) {{
                return xMap[String(d[xParam])] + offset + (Math.random() - 0.5) * 0.08;
            }});

            traces.push({{
                x: xNums,
                y: sub.map(function(d) {{ return d[yMetric]; }}),
                mode: 'markers', type: 'scatter',
                name: key, showlegend: true,
                marker: {{ size: 6, color: colorPalette[ci % colorPalette.length], opacity: 0.6,
                           line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }} }},
                text: sub.map(function(d) {{ return makeHover(d, yMetric); }}),
                hoverinfo: 'text',
                _uniqueX: uniqueX
            }});
        }});
    }}

    var allXRaw = filtered.map(function(d) {{ return String(d[xParam]); }});
    var uniqueX = [...new Set(allXRaw)];
    if (xParam !== 'activation') {{
        uniqueX.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
    }} else {{ uniqueX.sort(); }}

    var xAxisConfig;
    if (useColor) {{
        xAxisConfig = {{
            title: xAxisLabels[xParam],
            tickvals: uniqueX.map(function(v, i) {{ return i; }}),
            ticktext: uniqueX,
            zeroline: false
        }};
    }} else {{
        xAxisConfig = {{
            title: xAxisLabels[xParam], type: 'category',
            categoryorder: 'array', categoryarray: uniqueX
        }};
    }}

    var layout = {{
        xaxis: xAxisConfig,
        yaxis: {{ title: yMetric }},
        height: 550, margin: {{ t: 30, b: 60, l: 70, r: 30 }},
        hovermode: 'closest',
        showlegend: useColor,
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
