"""
V10 Unified Plot: Scatter / Boxplot / Violin in one HTML.
- Toggle between plot types via 3 radio-style buttons
- Data 1 / Data 2 metric selectors (Data 2 overlays a second metric, hides Color-by)
- Y Label is editable text independent of data selection
- Color-by dropdown (hidden when Data 2 is active — color distinguishes the two datasets)
- Boxmean toggle (diamond mean+SD) for box plots only
- X-axis selector, filter dropdowns, Models(N), Missing(N)
"""

from typing import Optional
import json as _json
import pandas as pd

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_unified_v10(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[str]:
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

    # Data 1 options (first metric selected)
    data1_options = ''
    for i, mc in enumerate(metric_cols):
        sel = ' selected' if i == 0 else ''
        data1_options += f'<option value="{mc}"{sel}>{mc}</option>\n'

    # Data 2 options (second metric selected if available, else first)
    data2_options = ''
    for i, mc in enumerate(metric_cols):
        sel = ' selected' if i == (1 if len(metric_cols) > 1 else 0) else ''
        data2_options += f'<option value="{mc}"{sel}>{mc}</option>\n'

    html = f"""<!DOCTYPE html>
<html><head>
<title>Explorer: {metric_column}</title>
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
    .filter-group input {{ font-size: 13px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; width: 100px; }}
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
    .data1-color {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px;
                    background: #52a8ad; margin-right: 2px; vertical-align: middle; }}
    .data2-color {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px;
                    background: #e05d5d; margin-right: 2px; vertical-align: middle; }}
    #plotDiv {{ width: 100%; }}
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
            <label>Y Label:</label>
            <input id="yLabel" type="text" value="{metric_column}" onchange="rebuild()" />
        </div>
        <div class="separator"></div>
        <div class="filter-group">
            <label><span class="data1-color"></span>Data 1:</label>
            <select id="data1" onchange="onData1Change()">
                {data1_options}
            </select>
        </div>
        <button id="data2Btn" class="btn" onclick="toggleData2()">Data 2: OFF</button>
        <div class="filter-group" id="data2Group" style="display:none;">
            <label><span class="data2-color"></span>Data 2:</label>
            <select id="data2" onchange="rebuild()">
                {data2_options}
            </select>
        </div>
        <div class="filter-group" id="colorByGroup">
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
<div id="plotDiv"></div>

<script>
var allData = {data_json};
var metricCols = {metric_cols_json};
var plotType = 'scatter';
var data2Active = false;
var showBoxmean = true;
var showLegend = false;

var DATA1_COLOR = '#52a8ad';
var DATA2_COLOR = '#e05d5d';

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
    document.getElementById('boxmeanBtn').style.display = (type === 'box') ? '' : 'none';
    rebuild();
}}

function toggleData2() {{
    data2Active = !data2Active;
    var btn = document.getElementById('data2Btn');
    btn.textContent = data2Active ? 'Data 2: ON' : 'Data 2: OFF';
    btn.classList.toggle('active', data2Active);
    document.getElementById('data2Group').style.display = data2Active ? '' : 'none';
    // Hide color-by when data2 is active (color is used for data1 vs data2)
    document.getElementById('colorByGroup').style.display = data2Active ? 'none' : '';
    if (data2Active) {{
        document.getElementById('colorBy').value = 'none';
    }}
    rebuild();
}}

function onData1Change() {{
    // Auto-update Y label to match Data 1 selection
    document.getElementById('yLabel').value = document.getElementById('data1').value;
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
    Plotly.relayout('plotDiv', {{'showlegend': showLegend}});
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

function makeHover(d, metric) {{
    return 'Act: ' + d.activation + '<br>Layers: ' + d.num_layers +
           '<br>Neurons: ' + d.neurons + '<br>LT: ' + d.leadtime + 'h' +
           '<br>Cycle: ' + d.cycle + '<br>' + metric + ': ' +
           (d[metric] != null ? d[metric].toFixed(4) : 'N/A');
}}

// Build traces for a single metric with optional color-by grouping
function buildSingleTraces(filtered, metric, xParam, colorBy, baseColor) {{
    var useColor = (colorBy !== 'none' && colorBy !== xParam);
    var traces = [];

    if (plotType === 'scatter') {{
        if (!useColor) {{
            traces.push({{
                x: filtered.map(function(d) {{ return String(d[xParam]); }}),
                y: filtered.map(function(d) {{ return d[metric]; }}),
                mode: 'markers', type: 'scatter',
                name: metric,
                marker: {{ size: 6, color: baseColor, opacity: 0.55,
                           line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }} }},
                text: filtered.map(function(d) {{ return makeHover(d, metric); }}),
                hoverinfo: 'text', showlegend: true
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
                    y: sub.map(function(d) {{ return d[metric]; }}),
                    mode: 'markers', type: 'scatter',
                    name: key, showlegend: true,
                    marker: {{ size: 6, color: colorPalette[ci % colorPalette.length], opacity: 0.6,
                               line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }} }},
                    text: sub.map(function(d) {{ return makeHover(d, metric); }}),
                    hoverinfo: 'text',
                    _uniqueX: uniqueX
                }});
            }});
        }}
    }} else {{
        // Box or Violin
        var traceType = (plotType === 'box') ? 'box' : 'violin';

        if (!useColor) {{
            var groups = {{}};
            filtered.forEach(function(d) {{
                var key = String(d[xParam]);
                if (!groups[key]) groups[key] = [];
                groups[key].push(d[metric]);
            }});
            var sortedKeys = Object.keys(groups);
            if (xParam !== 'activation') {{
                sortedKeys.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
            }} else {{ sortedKeys.sort(); }}

            sortedKeys.forEach(function(key, i) {{
                var t = {{
                    y: groups[key], type: traceType, name: metric + ' (' + key + ')',
                    legendgroup: metric,
                    marker: {{ color: baseColor, size: 3, opacity: 0.5 }},
                }};
                if (traceType === 'box') {{ t.boxmean = showBoxmean ? 'sd' : false; }}
                if (traceType === 'violin') {{
                    t.box = {{ visible: true }};
                    t.meanline = {{ visible: true }};
                    t.line = {{ color: baseColor }};
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
                    y: sub.map(function(d) {{ return d[metric]; }}),
                    type: traceType, name: cv,
                    marker: {{ color: colorPalette[ci % colorPalette.length], size: 3, opacity: 0.5 }},
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

// Build traces for dual-data mode (two metrics overlaid, distinguished by color)
function buildDualTraces(filtered, metric1, metric2, xParam) {{
    var traces = [];

    if (plotType === 'scatter') {{
        // Data 1
        traces.push({{
            x: filtered.map(function(d) {{ return String(d[xParam]); }}),
            y: filtered.map(function(d) {{ return d[metric1]; }}),
            mode: 'markers', type: 'scatter',
            name: metric1,
            marker: {{ size: 6, color: DATA1_COLOR, opacity: 0.55,
                       line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }} }},
            text: filtered.map(function(d) {{ return makeHover(d, metric1); }}),
            hoverinfo: 'text', showlegend: true
        }});
        // Data 2
        traces.push({{
            x: filtered.map(function(d) {{ return String(d[xParam]); }}),
            y: filtered.map(function(d) {{ return d[metric2]; }}),
            mode: 'markers', type: 'scatter',
            name: metric2,
            marker: {{ size: 6, color: DATA2_COLOR, opacity: 0.55,
                       line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }} }},
            text: filtered.map(function(d) {{ return makeHover(d, metric2); }}),
            hoverinfo: 'text', showlegend: true
        }});
    }} else {{
        var traceType = (plotType === 'box') ? 'box' : 'violin';

        // Group by x-param for both metrics
        var groups1 = {{}};
        var groups2 = {{}};
        filtered.forEach(function(d) {{
            var key = String(d[xParam]);
            if (!groups1[key]) groups1[key] = [];
            if (!groups2[key]) groups2[key] = [];
            if (d[metric1] != null) groups1[key].push(d[metric1]);
            if (d[metric2] != null) groups2[key].push(d[metric2]);
        }});

        var sortedKeys = Object.keys(groups1);
        if (xParam !== 'activation') {{
            sortedKeys.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
        }} else {{ sortedKeys.sort(); }}

        // Data 1 traces
        sortedKeys.forEach(function(key) {{
            var t = {{
                x: groups1[key].map(function() {{ return key; }}),
                y: groups1[key], type: traceType,
                name: metric1, legendgroup: metric1,
                showlegend: (key === sortedKeys[0]),
                marker: {{ color: DATA1_COLOR, size: 3, opacity: 0.5 }},
            }};
            if (traceType === 'box') {{ t.boxmean = showBoxmean ? 'sd' : false; }}
            if (traceType === 'violin') {{
                t.box = {{ visible: true }};
                t.meanline = {{ visible: true }};
                t.line = {{ color: DATA1_COLOR }};
                t.side = 'negative';
                t.scalegroup = key;
            }}
            traces.push(t);
        }});

        // Data 2 traces
        sortedKeys.forEach(function(key) {{
            var t = {{
                x: groups2[key].map(function() {{ return key; }}),
                y: groups2[key], type: traceType,
                name: metric2, legendgroup: metric2,
                showlegend: (key === sortedKeys[0]),
                marker: {{ color: DATA2_COLOR, size: 3, opacity: 0.5 }},
            }};
            if (traceType === 'box') {{ t.boxmean = showBoxmean ? 'sd' : false; }}
            if (traceType === 'violin') {{
                t.box = {{ visible: true }};
                t.meanline = {{ visible: true }};
                t.line = {{ color: DATA2_COLOR }};
                t.side = 'positive';
                t.scalegroup = key;
            }}
            traces.push(t);
        }});
    }}

    return {{ traces: traces, useColor: false }};
}}

function rebuild() {{
    var xParam = document.getElementById('xAxis').value;
    var yLabel = document.getElementById('yLabel').value;
    var data1Metric = document.getElementById('data1').value;
    var colorBy = document.getElementById('colorBy').value;

    var paramFiltered = applyFilter();

    // Filter to rows that have data1 metric
    var filtered = paramFiltered.filter(function(d) {{ return d[data1Metric] != null; }});
    var missingCount = paramFiltered.length - filtered.length;

    document.getElementById('modelCount').textContent = 'Models (' + filtered.length + ')';
    var missingEl = document.getElementById('missingCount');
    if (missingCount > 0) {{
        missingEl.textContent = 'Missing ' + data1Metric + ' (' + missingCount + ')';
        missingEl.style.display = '';
    }} else {{
        missingEl.style.display = 'none';
    }}

    var result;
    var legendTitle = '';

    if (data2Active) {{
        var data2Metric = document.getElementById('data2').value;
        // Also filter to rows that have data2 metric
        filtered = filtered.filter(function(d) {{ return d[data2Metric] != null; }});
        result = buildDualTraces(filtered, data1Metric, data2Metric, xParam);
        legendTitle = 'Metric';
    }} else {{
        result = buildSingleTraces(filtered, data1Metric, xParam, colorBy, DATA1_COLOR);
        legendTitle = xAxisLabels[colorBy] || '';
    }}

    // Build layout
    var allXRaw = filtered.map(function(d) {{ return String(d[xParam]); }});
    var uniqueX = [...new Set(allXRaw)];
    if (xParam !== 'activation') {{
        uniqueX.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
    }} else {{ uniqueX.sort(); }}

    var xAxisConfig;
    if (plotType === 'scatter' && result.useColor) {{
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
    var layout = {{
        xaxis: xAxisConfig,
        yaxis: {{ title: yLabel, rangemode: (plotType === 'scatter') ? 'normal' : 'tozero' }},
        height: h, margin: {{ t: 30, b: 60, l: 70, r: 30 }},
        boxmode: 'group', violinmode: 'overlay',
        hovermode: 'closest',
        showlegend: showLegend || data2Active,
        legend: {{ title: {{ text: legendTitle }} }}
    }};

    Plotly.newPlot('plotDiv', result.traces, layout, {{ responsive: true }});
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
