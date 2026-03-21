"""
V9 Unified Plot: Scatter / Boxplot / Violin in one HTML.
- Toggle between plot types via 3 radio-style buttons
- Y-axis metric dropdown (works for all 3 plot types)
- Dual mode: side-by-side with two independently selectable color-by options
- Boxmean toggle (diamond mean+SD) for box plots
- X-axis selector, color-by, filter dropdowns, Models(N), Missing(N)
"""

from typing import Optional
import json as _json
import pandas as pd

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_unified_v9(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[str]:
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()

    # Detect all numeric metric columns
    _skip = {'run_id', 'iteration', 'leadtime', 'cycle', 'num_layers', 'neurons'}
    metric_cols = [c for c in clean.columns
                   if c not in _skip
                   and pd.api.types.is_numeric_dtype(clean[c])
                   and clean[c].notna().any()
                   and c not in ('metric_type',)]
    if metric_column in metric_cols:
        metric_cols.remove(metric_column)
    metric_cols.insert(0, metric_column)

    # Build records with ALL metric columns
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
    metric_cols_json = _json.dumps(metric_cols)

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

    y_axis_options = ''
    for i, mc in enumerate(metric_cols):
        sel = ' selected' if i == 0 else ''
        y_axis_options += f'<option value="{mc}"{sel}>{mc}</option>\n'

    html = f"""<!DOCTYPE html>
<html><head>
<title>Unified Plot: {metric_column}</title>
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
    .missing-count {{ font-size: 13px; font-weight: 600; color: #c0392b; padding: 4px 10px;
                      background: rgba(192, 57, 43, 0.1); border-radius: 4px; }}
    .toggle-btn {{ font-size: 13px; padding: 5px 14px; border: 1px solid #ccc; border-radius: 4px;
                   background: white; cursor: pointer; font-weight: 600; }}
    .toggle-btn:hover {{ background: #e9ecef; }}
    .toggle-btn.active {{ background: #52a8ad; color: white; border-color: #52a8ad; }}
    .btn {{ font-size: 12px; padding: 5px 12px; border: 1px solid #ccc; border-radius: 4px;
            background: white; cursor: pointer; }}
    .btn:hover {{ background: #e9ecef; }}
    .btn.active {{ background: #52a8ad; color: white; border-color: #52a8ad; }}
    .plot-container {{ display: flex; gap: 8px; }}
    .plot-container > div {{ flex: 1; }}
    #plotRight {{ display: none; }}
</style>
</head><body>
<div class="controls">
    <h2>Hyperparameter Explorer</h2>
    <div class="filter-bar">
        <!-- Plot type toggle -->
        <div class="filter-group">
            <button class="toggle-btn active" id="btnScatter" onclick="setPlotType('scatter')">Scatter</button>
            <button class="toggle-btn" id="btnBox" onclick="setPlotType('box')">Box</button>
            <button class="toggle-btn" id="btnViolin" onclick="setPlotType('violin')">Violin</button>
        </div>
        <div class="separator"></div>
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
        <button id="dualBtn" class="btn" onclick="toggleDual()">Dual: OFF</button>
        <div class="filter-group" id="colorByRightGroup" style="display:none;">
            <label>Color Right:</label>
            <select id="colorByRight" onchange="rebuild()">
                <option value="none" selected>None</option>
                <option value="activation">Activation</option>
                <option value="num_layers">Num Layers</option>
                <option value="neurons">Hidden Units</option>
                <option value="leadtime">Leadtime</option>
                <option value="cycle">Cycle</option>
            </select>
        </div>
        <button id="pointsBtn" class="btn" style="display:none;" onclick="togglePoints()">Points: OFF</button>
        <button id="boxmeanBtn" class="btn" style="display:none;" onclick="toggleBoxmean()">Mean+SD: ON</button>
        <button id="legendBtn" class="btn" onclick="toggleLegend()">Legend: OFF</button>
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
        <span id="missingCount" class="missing-count" style="display:none;">Missing (0)</span>
    </div>
</div>
<div class="plot-container">
    <div id="plotLeft"></div>
    <div id="plotRight"></div>
</div>

<script>
var allData = {data_json};
var metricCols = {metric_cols_json};
var plotType = 'scatter';
var dualMode = false;
var showPoints = false;
var showBoxmean = true;
var showLegend = false;

var xAxisLabels = {{
    'neurons': 'Hidden Units', 'num_layers': 'Number of Layers',
    'leadtime': 'Leadtime (h)', 'cycle': 'Cycle', 'activation': 'Activation Function'
}};

var colorPalette = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
    '#a65628','#f781bf','#66c2a5','#fc8d62','#8da0cb',
    '#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'
];

function setPlotType(type) {{
    plotType = type;
    document.getElementById('btnScatter').classList.toggle('active', type === 'scatter');
    document.getElementById('btnBox').classList.toggle('active', type === 'box');
    document.getElementById('btnViolin').classList.toggle('active', type === 'violin');
    document.getElementById('pointsBtn').style.display = (type === 'scatter') ? 'none' : '';
    document.getElementById('boxmeanBtn').style.display = (type === 'box') ? '' : 'none';
    rebuild();
}}

function toggleDual() {{
    dualMode = !dualMode;
    var btn = document.getElementById('dualBtn');
    btn.textContent = dualMode ? 'Dual: ON' : 'Dual: OFF';
    btn.classList.toggle('active', dualMode);
    document.getElementById('colorByRightGroup').style.display = dualMode ? '' : 'none';
    document.getElementById('plotRight').style.display = dualMode ? '' : 'none';
    rebuild();
}}

function togglePoints() {{
    showPoints = !showPoints;
    var btn = document.getElementById('pointsBtn');
    btn.textContent = showPoints ? 'Points: ON' : 'Points: OFF';
    btn.classList.toggle('active', showPoints);
    rebuild();
}}

function toggleBoxmean() {{
    showBoxmean = !showBoxmean;
    var btn = document.getElementById('boxmeanBtn');
    btn.textContent = showBoxmean ? 'Mean+SD: ON' : 'Mean+SD: OFF';
    btn.classList.toggle('active', showBoxmean);
    rebuild();
}}

function toggleLegend() {{
    showLegend = !showLegend;
    var btn = document.getElementById('legendBtn');
    btn.textContent = showLegend ? 'Legend: ON' : 'Legend: OFF';
    btn.classList.toggle('active', showLegend);
    Plotly.relayout('plotLeft', {{'showlegend': showLegend}});
    if (dualMode) Plotly.relayout('plotRight', {{'showlegend': showLegend}});
}}

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

function makeHover(d, yMetric) {{
    return 'Act: ' + d.activation + '<br>Layers: ' + d.num_layers +
           '<br>Neurons: ' + d.neurons + '<br>LT: ' + d.leadtime + 'h' +
           '<br>Cycle: ' + d.cycle + '<br>' + yMetric + ': ' +
           (d[yMetric] != null ? d[yMetric].toFixed(4) : 'N/A');
}}

function buildTraces(filtered, yMetric, xParam, colorBy) {{
    var useColor = (colorBy !== 'none' && colorBy !== xParam);
    var traces = [];

    if (plotType === 'scatter') {{
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

            var allXRaw = filtered.map(function(d) {{ return String(d[xParam]); }});
            var uniqueX = [...new Set(allXRaw)];
            if (xParam !== 'activation') {{
                uniqueX.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
            }} else {{ uniqueX.sort(); }}
            var xMap = {{}};
            uniqueX.forEach(function(v, i) {{ xMap[v] = i; }});

            keys.forEach(function(key, ci) {{
                var sub = groups[key];
                var offset = startOffset + ci * step;
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
    }} else {{
        // Box or Violin
        var traceType = (plotType === 'box') ? 'box' : 'violin';
        var pointsMode = showPoints ? 'all' : false;

        if (!useColor) {{
            var groups = {{}};
            filtered.forEach(function(d) {{
                var key = String(d[xParam]);
                if (!groups[key]) groups[key] = [];
                groups[key].push(d[yMetric]);
            }});
            var sortedKeys = Object.keys(groups);
            if (xParam !== 'activation') {{
                sortedKeys.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
            }} else {{ sortedKeys.sort(); }}

            sortedKeys.forEach(function(key, i) {{
                var t = {{
                    y: groups[key], type: traceType, name: key,
                    marker: {{ color: colorPalette[i % colorPalette.length], size: 3, opacity: 0.5 }},
                    points: pointsMode, boxpoints: pointsMode,
                    jitter: 0.5, pointpos: 0.8
                }};
                if (traceType === 'box') {{ t.boxmean = showBoxmean ? 'sd' : false; }}
                if (traceType === 'violin') {{
                    t.box = {{ visible: true }};
                    t.meanline = {{ visible: true }};
                    t.line = {{ color: colorPalette[i % colorPalette.length] }};
                }}
                traces.push(t);
            }});
        }} else {{
            var colorVals = [...new Set(filtered.map(function(d) {{ return String(d[colorBy]); }}))];
            if (colorBy !== 'activation') {{
                colorVals.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
            }} else {{ colorVals.sort(); }}

            colorVals.forEach(function(cv, ci) {{
                var sub = filtered.filter(function(d) {{ return String(d[colorBy]) === cv; }});
                var t = {{
                    x: sub.map(function(d) {{ return String(d[xParam]); }}),
                    y: sub.map(function(d) {{ return d[yMetric]; }}),
                    type: traceType, name: cv,
                    marker: {{ color: colorPalette[ci % colorPalette.length], size: 3, opacity: 0.5 }},
                    points: pointsMode, boxpoints: pointsMode,
                    jitter: 0.5, pointpos: 0.8
                }};
                if (traceType === 'box') {{ t.boxmean = showBoxmean ? 'sd' : false; }}
                if (traceType === 'violin') {{
                    t.box = {{ visible: true }};
                    t.meanline = {{ visible: true }};
                    t.line = {{ color: colorPalette[ci % colorPalette.length] }};
                }}
                traces.push(t);
            }});
        }}
    }}

    return {{ traces: traces, useColor: useColor }};
}}

function buildLayout(yMetric, xParam, colorBy, useColor, filtered) {{
    var allXRaw = filtered.map(function(d) {{ return String(d[xParam]); }});
    var uniqueX = [...new Set(allXRaw)];
    if (xParam !== 'activation') {{
        uniqueX.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
    }} else {{ uniqueX.sort(); }}

    var xAxisConfig;
    if (plotType === 'scatter' && useColor) {{
        xAxisConfig = {{
            title: xAxisLabels[xParam],
            tickvals: uniqueX.map(function(v, i) {{ return i; }}),
            ticktext: uniqueX, zeroline: false
        }};
    }} else {{
        xAxisConfig = {{
            title: xAxisLabels[xParam], type: 'category',
            categoryorder: 'array', categoryarray: uniqueX
        }};
    }}

    var h = Math.max(500, window.innerHeight - 160);
    return {{
        xaxis: xAxisConfig,
        yaxis: {{ title: yMetric, rangemode: (plotType === 'scatter') ? 'normal' : 'tozero' }},
        height: h, margin: {{ t: 30, b: 60, l: 70, r: 30 }},
        boxmode: 'group', violinmode: 'group',
        hovermode: 'closest',
        showlegend: showLegend,
        legend: {{ title: {{ text: xAxisLabels[colorBy] || '' }} }}
    }};
}}

function rebuild() {{
    var xParam = document.getElementById('xAxis').value;
    var yMetric = document.getElementById('yAxis').value;
    var colorBy = document.getElementById('colorBy').value;

    var paramFiltered = applyFilter();
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

    var result = buildTraces(filtered, yMetric, xParam, colorBy);
    var layout = buildLayout(yMetric, xParam, colorBy, result.useColor, filtered);
    Plotly.newPlot('plotLeft', result.traces, layout, {{ responsive: true }});

    if (dualMode) {{
        var colorByR = document.getElementById('colorByRight').value;
        var resultR = buildTraces(filtered, yMetric, xParam, colorByR);
        var layoutR = buildLayout(yMetric, xParam, colorByR, resultR.useColor, filtered);
        Plotly.newPlot('plotRight', resultR.traces, layoutR, {{ responsive: true }});
    }}
}}

window.addEventListener('resize', function() {{ rebuild(); }});
rebuild();
</script>
</body></html>"""

    if output_dir:
        with open(f"{output_dir}/01_explorer.html", 'w') as f:
            f.write(html)
        print("Saved: 01_explorer.html")
    return html
