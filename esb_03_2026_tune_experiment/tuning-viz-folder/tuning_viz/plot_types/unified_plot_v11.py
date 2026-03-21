"""
V11 Unified Plot: Scatter / Boxplot / Violin in one HTML.
- Toggle between plot types via 3 radio-style buttons
- Dynamic data series: add/remove metric overlays with [+] / [x] buttons
- Each data series gets a distinct color; scatter points are jittered apart per series
- Color-by dropdown (hidden when multiple data series active)
- Y Label is editable text independent of data selection
- Boxmean toggle for box plots only
- Legend hidden by default
"""

from typing import Optional
import json as _json
import pandas as pd

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_unified_v11(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[str]:
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()

    _skip = {'run_id', 'iteration', 'leadtime', 'cycle', 'num_layers', 'neurons'}
    metric_cols = [c for c in clean.columns
                   if c not in _skip
                   and pd.api.types.is_numeric_dtype(clean[c])
                   and clean[c].notna().any()
                   and c not in ('metric_type',)]
    if metric_column in metric_cols:
        metric_cols.remove(metric_column)
    metric_cols.insert(0, metric_column)

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

    metric_options_html = ''
    for i, mc in enumerate(metric_cols):
        metric_options_html += f'<option value="{mc}">{mc}</option>\n'

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
    .data-series {{ display: flex; align-items: center; gap: 4px; }}
    .data-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: middle; }}
    .remove-btn {{ font-size: 11px; padding: 2px 6px; border: 1px solid #ccc; border-radius: 3px;
                   background: white; cursor: pointer; color: #c0392b; font-weight: bold; }}
    .remove-btn:hover {{ background: #fce4e4; }}
    .add-btn {{ font-size: 13px; padding: 3px 10px; border: 1px solid #aaa; border-radius: 4px;
                background: white; cursor: pointer; font-weight: bold; color: #333; }}
    .add-btn:hover {{ background: #e9ecef; }}
    #plotDiv {{ width: 100%; }}
</style>
</head><body>
<div class="controls">
    <h2>Hyperparameter Explorer</h2>
    <div class="filter-bar">
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
        <!-- Dynamic data series container -->
        <div id="dataSeriesContainer" style="display:flex; flex-wrap:wrap; align-items:center; gap:8px;"></div>
        <button class="add-btn" onclick="addDataSeries()" title="Add another metric">+</button>
        <div class="separator"></div>
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
var metricOptionsHTML = '{metric_options_html.replace(chr(10), "").replace(chr(13), "")}';
var plotType = 'scatter';
var showBoxmean = true;
var showLegend = false;

// Each data series: {{ id: unique int, metric: string }}
var dataSeriesCounter = 0;
var dataSeries = [];

var seriesColors = [
    '#52a8ad', '#e05d5d', '#4daf4a', '#984ea3', '#ff7f00',
    '#377eb8', '#a65628', '#f781bf', '#66c2a5', '#fc8d62'
];

var xAxisLabels = {{
    'neurons': 'Hidden Units', 'num_layers': 'Number of Layers',
    'leadtime': 'Leadtime (h)', 'cycle': 'Cycle', 'activation': 'Activation Function'
}};

var colorPalette = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
    '#a65628','#f781bf','#66c2a5','#fc8d62','#8da0cb',
    '#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'
];

function getSeriesColor(index) {{
    return seriesColors[index % seriesColors.length];
}}

function renderDataSeries() {{
    var container = document.getElementById('dataSeriesContainer');
    container.innerHTML = '';
    dataSeries.forEach(function(s, idx) {{
        var color = getSeriesColor(idx);
        var div = document.createElement('div');
        div.className = 'data-series';
        var showRemove = dataSeries.length > 1;
        div.innerHTML =
            '<span class="data-dot" style="background:' + color + '"></span>' +
            '<select onchange="updateSeriesMetric(' + s.id + ', this.value)" style="font-size:12px; padding:3px 6px; border:1px solid #ccc; border-radius:4px;">' +
            metricOptionsHTML.replace('value="' + s.metric + '"', 'value="' + s.metric + '" selected') +
            '</select>' +
            (showRemove ? '<button class="remove-btn" onclick="removeDataSeries(' + s.id + ')" title="Remove">&times;</button>' : '');
        container.appendChild(div);
    }});
    updateColorByVisibility();
}}

function addDataSeries() {{
    // Pick the next metric that isn't already selected, or default to first
    var usedMetrics = dataSeries.map(function(s) {{ return s.metric; }});
    var nextMetric = metricCols.find(function(m) {{ return usedMetrics.indexOf(m) === -1; }}) || metricCols[0];
    dataSeriesCounter++;
    dataSeries.push({{ id: dataSeriesCounter, metric: nextMetric }});
    renderDataSeries();
    rebuild();
}}

function removeDataSeries(id) {{
    dataSeries = dataSeries.filter(function(s) {{ return s.id !== id; }});
    renderDataSeries();
    rebuild();
}}

function updateSeriesMetric(id, metric) {{
    dataSeries.forEach(function(s) {{
        if (s.id === id) s.metric = metric;
    }});
    // Auto-update Y label if only one series
    if (dataSeries.length === 1) {{
        document.getElementById('yLabel').value = metric;
    }}
    rebuild();
}}

function updateColorByVisibility() {{
    var multi = dataSeries.length > 1;
    document.getElementById('colorByGroup').style.display = multi ? 'none' : '';
    if (multi) {{
        document.getElementById('colorBy').value = 'none';
    }}
}}

function setPlotType(type) {{
    plotType = type;
    document.getElementById('btnScatter').classList.toggle('active', type === 'scatter');
    document.getElementById('btnBox').classList.toggle('active', type === 'box');
    document.getElementById('btnViolin').classList.toggle('active', type === 'violin');
    document.getElementById('boxmeanBtn').style.display = (type === 'box') ? '' : 'none';
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

function buildMultiSeriesTraces(filtered, metrics, xParam) {{
    var nMetrics = metrics.length;
    var traces = [];

    if (plotType === 'scatter') {{
        // Build x-position map for jitter offsets
        var allXRaw = filtered.map(function(d) {{ return String(d[xParam]); }});
        var uniqueX = [...new Set(allXRaw)];
        if (xParam !== 'activation') {{
            uniqueX.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
        }} else {{ uniqueX.sort(); }}
        var xMap = {{}};
        uniqueX.forEach(function(v, i) {{ xMap[v] = i; }});

        var spread = 0.6;
        var step = nMetrics > 1 ? spread / (nMetrics - 1) : 0;
        var startOffset = nMetrics > 1 ? -spread / 2 : 0;

        metrics.forEach(function(metric, mi) {{
            var color = getSeriesColor(mi);
            var offset = startOffset + mi * step;
            var validData = filtered.filter(function(d) {{ return d[metric] != null; }});

            if (nMetrics === 1) {{
                // Single series: use category x directly (no numeric mapping)
                traces.push({{
                    x: validData.map(function(d) {{ return String(d[xParam]); }}),
                    y: validData.map(function(d) {{ return d[metric]; }}),
                    mode: 'markers', type: 'scatter',
                    name: metric,
                    marker: {{ size: 6, color: color, opacity: 0.55,
                               line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }} }},
                    text: validData.map(function(d) {{ return makeHover(d, metric); }}),
                    hoverinfo: 'text', showlegend: true
                }});
            }} else {{
                // Multi series: use numeric x with offsets
                var xNums = validData.map(function(d) {{
                    return xMap[String(d[xParam])] + offset + (Math.random() - 0.5) * 0.08;
                }});
                traces.push({{
                    x: xNums,
                    y: validData.map(function(d) {{ return d[metric]; }}),
                    mode: 'markers', type: 'scatter',
                    name: metric,
                    marker: {{ size: 6, color: color, opacity: 0.6,
                               line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }} }},
                    text: validData.map(function(d) {{ return makeHover(d, metric); }}),
                    hoverinfo: 'text', showlegend: true,
                    _uniqueX: uniqueX
                }});
            }}
        }});
    }} else {{
        var traceType = (plotType === 'box') ? 'box' : 'violin';

        // Group by x-param
        var allXRaw = filtered.map(function(d) {{ return String(d[xParam]); }});
        var uniqueX = [...new Set(allXRaw)];
        if (xParam !== 'activation') {{
            uniqueX.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
        }} else {{ uniqueX.sort(); }}

        metrics.forEach(function(metric, mi) {{
            var color = getSeriesColor(mi);

            var groups = {{}};
            filtered.forEach(function(d) {{
                var key = String(d[xParam]);
                if (!groups[key]) groups[key] = [];
                if (d[metric] != null) groups[key].push(d[metric]);
            }});

            if (nMetrics === 1) {{
                // Single metric: one trace per x-group
                uniqueX.forEach(function(key, i) {{
                    var vals = groups[key] || [];
                    var t = {{
                        y: vals, type: traceType, name: key,
                        marker: {{ color: colorPalette[i % colorPalette.length], size: 3, opacity: 0.5 }},
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
                // Multi metric: one trace per x-group per metric, grouped
                uniqueX.forEach(function(key) {{
                    var vals = groups[key] || [];
                    var t = {{
                        x: vals.map(function() {{ return key; }}),
                        y: vals, type: traceType,
                        name: metric, legendgroup: metric,
                        showlegend: (key === uniqueX[0]),
                        marker: {{ color: color, size: 3, opacity: 0.5 }},
                    }};
                    if (traceType === 'box') {{ t.boxmean = showBoxmean ? 'sd' : false; }}
                    if (traceType === 'violin') {{
                        t.box = {{ visible: true }};
                        t.meanline = {{ visible: true }};
                        t.line = {{ color: color }};
                        if (nMetrics === 2) {{
                            t.side = (mi === 0) ? 'negative' : 'positive';
                        }}
                        t.scalegroup = metric + '_' + key;
                    }}
                    traces.push(t);
                }});
            }}
        }});
    }}

    return traces;
}}

// Single metric with color-by grouping (unchanged from v10)
function buildColorByTraces(filtered, metric, xParam, colorBy) {{
    var traces = [];
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

    if (plotType === 'scatter') {{
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
    }} else {{
        var traceType = (plotType === 'box') ? 'box' : 'violin';
        keys.forEach(function(cv, ci) {{
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

    return traces;
}}

function rebuild() {{
    var xParam = document.getElementById('xAxis').value;
    var yLabel = document.getElementById('yLabel').value;
    var colorBy = document.getElementById('colorBy').value;
    var metrics = dataSeries.map(function(s) {{ return s.metric; }});
    var multiData = metrics.length > 1;

    var paramFiltered = applyFilter();
    // Filter to rows that have the first metric at least
    var filtered = paramFiltered.filter(function(d) {{ return d[metrics[0]] != null; }});
    var missingCount = paramFiltered.length - filtered.length;

    document.getElementById('modelCount').textContent = 'Models (' + filtered.length + ')';
    var missingEl = document.getElementById('missingCount');
    if (missingCount > 0) {{
        missingEl.textContent = 'Missing ' + metrics[0] + ' (' + missingCount + ')';
        missingEl.style.display = '';
    }} else {{
        missingEl.style.display = 'none';
    }}

    var traces;
    var useNumericX = false;
    var legendTitle = '';

    if (multiData || colorBy === 'none' || colorBy === xParam) {{
        traces = buildMultiSeriesTraces(filtered, metrics, xParam);
        useNumericX = multiData;
        legendTitle = multiData ? 'Metric' : '';
    }} else {{
        traces = buildColorByTraces(filtered, metrics[0], xParam, colorBy);
        useNumericX = true;
        legendTitle = xAxisLabels[colorBy] || '';
    }}

    // Build x-axis
    var allXRaw = filtered.map(function(d) {{ return String(d[xParam]); }});
    var uniqueX = [...new Set(allXRaw)];
    if (xParam !== 'activation') {{
        uniqueX.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
    }} else {{ uniqueX.sort(); }}

    var xAxisConfig;
    if (plotType === 'scatter' && useNumericX) {{
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
        showlegend: showLegend,
        legend: {{ title: {{ text: legendTitle }} }}
    }};

    Plotly.newPlot('plotDiv', traces, layout, {{ responsive: true }});
}}

// Initialize with one data series
dataSeriesCounter = 1;
dataSeries = [{{ id: 1, metric: metricCols[0] }}];
renderDataSeries();

window.addEventListener('resize', function() {{ rebuild(); }});
rebuild();
</script>
</body></html>"""

    if output_dir:
        with open(f"{output_dir}/01_explorer.html", 'w') as f:
            f.write(html)
        print("Saved: 01_explorer.html")
    return html
