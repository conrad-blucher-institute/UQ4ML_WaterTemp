"""
V18 Time Series Comparison: model picker + stacked Plotly charts + loss curves.

Changes from v17:
- Loss Curves chart: loss vs val_loss per epoch, loaded from _history.json
  in the keras_files/ sibling directory
- "Loss Curves" toggle button to show/hide the chart

Usage: run via serve_timeseries.py which starts a local HTTP server and
opens the HTML in a browser.
"""

from typing import Optional
from pathlib import Path
import json as _json
import os

import pandas as pd


def plot_timeseries_compare_v18(predictions_dir: str, output_dir: str) -> Optional[str]:
    """Generate the time series comparison HTML."""
    pred_dir = Path(predictions_dir)
    index_path = pred_dir / '_index.csv'
    if not index_path.exists():
        print("No _index.csv found in predictions/. Run generate_predictions.py first.")
        return None

    index_df = pd.read_csv(index_path)
    if len(index_df) == 0:
        print("_index.csv is empty.")
        return None

    # Collect all metric columns (numeric, non-parameter columns)
    param_cols = {'model_name', 'activation', 'num_layers', 'neurons', 'leadtime', 'cycle', 'run_num'}
    metric_cols = [c for c in index_df.columns
                   if c not in param_cols
                   and pd.api.types.is_numeric_dtype(index_df[c])
                   and index_df[c].notna().any()]
    # Ensure val_mae is first if present
    if 'val_mae' in metric_cols:
        metric_cols.remove('val_mae')
        metric_cols.insert(0, 'val_mae')

    index_records = []
    for _, row in index_df.iterrows():
        rec = {
            'model_name': str(row['model_name']),
            'activation': str(row.get('activation', '')),
            'num_layers': int(row.get('num_layers', 0)),
            'neurons': int(row.get('neurons', 0)),
            'leadtime': int(row.get('leadtime', 0)),
            'cycle': int(row.get('cycle', 0)),
        }
        for mc in metric_cols:
            rec[mc] = round(float(row[mc]), 4) if pd.notna(row.get(mc)) else None
        index_records.append(rec)

    filter_values = {
        'activation': sorted(index_df['activation'].dropna().unique().tolist()),
        'num_layers': sorted(index_df['num_layers'].dropna().unique().astype(int).tolist()),
        'neurons': sorted(index_df['neurons'].dropna().unique().astype(int).tolist()),
        'leadtime': sorted(index_df['leadtime'].dropna().unique().astype(int).tolist()),
        'cycle': sorted(index_df['cycle'].dropna().unique().astype(int).tolist()),
    }

    out_path = Path(output_dir).resolve()
    pred_path = pred_dir.resolve()
    try:
        rel_pred = os.path.relpath(pred_path, out_path)
        pred_url_prefix = rel_pred.replace('\\', '/')
    except ValueError:
        pred_url_prefix = str(pred_path).replace('\\', '/')

    # keras_files/ is a sibling of predictions/
    keras_path = pred_dir.resolve().parent / 'keras_files'
    try:
        rel_keras = os.path.relpath(keras_path, out_path)
        history_url_prefix = rel_keras.replace('\\', '/')
    except ValueError:
        history_url_prefix = str(keras_path).replace('\\', '/')

    has_keras_files = keras_path.exists()

    index_json = _json.dumps(index_records)
    filters_json = _json.dumps(filter_values)
    metric_cols_json = _json.dumps(metric_cols)

    html = f"""<!DOCTYPE html>
<html><head>
<title>Time Series Comparison (v18)</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           display: flex; height: 100vh; overflow: hidden; }}
    .left-panel {{ width: 35%; min-width: 280px; max-width: 70%; border-right: 2px solid #ddd;
                   display: flex; flex-direction: column; background: #fafafa; position: relative; }}
    .resize-handle {{ position: absolute; top: 0; right: -4px; width: 8px; height: 100%;
                      cursor: col-resize; z-index: 200; background: transparent; }}
    .resize-handle:hover, .resize-handle.active {{ background: rgba(82, 168, 173, 0.3); }}
    .panel-header {{ padding: 12px 16px; background: #f0f0f0; border-bottom: 1px solid #ddd; }}
    .panel-header h2 {{ font-size: 16px; margin-bottom: 8px; display: flex; align-items: center; gap: 10px; }}
    .filter-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 8px; align-items: center; }}
    .filter-group {{ display: flex; align-items: center; gap: 4px; }}
    .filter-group label {{ font-size: 11px; font-weight: 600; color: #555; white-space: nowrap; }}
    .filter-group select {{ font-size: 11px; padding: 3px 6px; border: 1px solid #ccc; border-radius: 4px; }}
    .status-row {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
    .model-count {{ font-size: 12px; color: #888; }}
    .selected-count {{ font-size: 12px; font-weight: 600; color: #52a8ad; }}
    .loading-indicator {{ font-size: 11px; color: #ff7f00; font-weight: 600; display: none; }}
    .explorer-banner {{ font-size: 12px; padding: 6px 12px; background: #fff3e0; border-bottom: 1px solid #ffcc80;
                        display: none; align-items: center; gap: 8px; }}
    .explorer-banner.visible {{ display: flex; }}
    .explorer-banner .dismiss {{ cursor: pointer; color: #999; font-weight: bold; margin-left: auto; }}
    .explorer-banner .dismiss:hover {{ color: #333; }}
    .btn {{ font-size: 11px; padding: 3px 10px; border: 1px solid #ccc; border-radius: 4px;
            background: white; cursor: pointer; font-weight: 600; }}
    .btn:hover {{ background: #e9ecef; }}
    .btn.active {{ background: #52a8ad; color: white; border-color: #52a8ad; }}
    .btn-refresh {{ font-size: 11px; padding: 3px 10px; border: 1px solid #52a8ad; border-radius: 4px;
                    background: white; cursor: pointer; color: #52a8ad; font-weight: 600; }}
    .btn-refresh:hover {{ background: #52a8ad; color: white; }}
    .ms-wrap {{ position: relative; display: inline-block; }}
    .ms-btn {{ font-size: 11px; padding: 3px 6px; border: 1px solid #ccc; border-radius: 4px;
               background: white; cursor: pointer; min-width: 50px; text-align: left; }}
    .ms-btn:hover {{ background: #f5f5f5; }}
    .ms-panel {{ display: none; position: absolute; top: 100%; left: 0; z-index: 100;
                 background: white; border: 1px solid #ccc; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                 padding: 6px; min-width: 80px; max-height: 200px; overflow-y: auto; }}
    .ms-panel.open {{ display: block; }}
    .ms-panel label {{ display: block; font-size: 11px; padding: 2px 4px; cursor: pointer; white-space: nowrap; }}
    .ms-panel label:hover {{ background: #f0f7f7; }}
    .ms-panel input {{ margin-right: 4px; }}
    .metric-checkboxes {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
    .metric-cb-label {{ display: flex; align-items: center; gap: 3px; font-size: 11px;
                        padding: 2px 6px; border: 1px solid #ddd; border-radius: 4px;
                        cursor: pointer; background: white; user-select: none; }}
    .metric-cb-label:hover {{ background: #f0f7f7; }}
    .metric-cb-label.checked {{ background: #e8f4f5; border-color: #52a8ad; }}
    .metric-cb-label input {{ margin: 0; }}
    .sort-label {{ font-size: 11px; font-weight: 600; color: #555; }}
    .table-wrap {{ flex: 1; overflow-y: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
    th {{ background: rgb(50, 60, 80); color: white; padding: 6px 4px; position: sticky;
         top: 0; cursor: pointer; white-space: nowrap; user-select: none; z-index: 2; }}
    th:hover {{ background: rgb(70, 80, 100); }}
    td {{ padding: 5px 4px; text-align: center; border-bottom: 1px solid #eee; }}
    tr:hover {{ background: #e8f4f5; }}
    tr.selected {{ background: #d4edee; }}
    .cb-cell {{ width: 30px; }}
    .color-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                  margin-right: 3px; vertical-align: middle; }}
    .right-panel {{ flex: 1; display: flex; flex-direction: column; padding: 8px; overflow: hidden; }}
    .chart-section {{ display: flex; flex-direction: column; min-height: 0; }}
    .chart-container {{ flex: 1; min-height: 0; }}
    .chart-header {{ display: flex; align-items: center; gap: 10px; padding: 4px 8px; }}
    .chart-label {{ font-size: 13px; font-weight: 600; color: #555; }}
    .no-selection {{ display: flex; align-items: center; justify-content: center;
                     height: 100%; color: #aaa; font-size: 16px; }}
    .toggle-row {{ display: flex; align-items: center; gap: 8px; padding: 4px 8px; flex-wrap: wrap; }}
    .toggle-btn {{ font-size: 11px; padding: 4px 12px; border: 1px solid #ccc; border-radius: 4px;
                   background: white; cursor: pointer; font-weight: 600; transition: all 0.15s; }}
    .toggle-btn:hover {{ background: #e9ecef; }}
    .toggle-btn.on {{ background: #52a8ad; color: white; border-color: #52a8ad; }}
    .toggle-btn.loss-btn {{ border-color: #d35400; color: #d35400; }}
    .toggle-btn.loss-btn:hover {{ background: #fdf2e9; }}
    .toggle-btn.loss-btn.on {{ background: #d35400; color: white; border-color: #d35400; }}
    .toggle-btn.legend-btn {{ border-color: #8e8e8e; }}
    .toggle-btn.legend-btn.on {{ background: #6c757d; color: white; border-color: #6c757d; }}
    .metric-vals-toggle {{ font-size: 11px; padding: 4px 12px; border: 1px solid #e6a817; border-radius: 4px;
                           background: white; cursor: pointer; font-weight: 600; transition: all 0.15s; color: #e6a817; }}
    .metric-vals-toggle:hover {{ background: #fef9e7; }}
    .metric-vals-toggle.on {{ background: #e6a817; color: white; border-color: #e6a817; }}
</style>
</head><body>

<div class="left-panel" id="leftPanel">
    <div class="panel-header">
        <h2>Model Picker
            <button class="btn-refresh" onclick="refreshFromExplorer()" title="Reload models sent from Explorer">Refresh</button>
        </h2>
        <div class="filter-row">
            <div class="filter-group">
                <label>Activation:</label>
                <div class="ms-wrap" id="msAct"></div>
            </div>
            <div class="filter-group">
                <label>Layers:</label>
                <div class="ms-wrap" id="msLayers"></div>
            </div>
            <div class="filter-group">
                <label>Neurons:</label>
                <div class="ms-wrap" id="msNeurons"></div>
            </div>
            <div class="filter-group">
                <label>LT:</label>
                <div class="ms-wrap" id="msLT"></div>
            </div>
            <div class="filter-group">
                <label>Cycle:</label>
                <div class="ms-wrap" id="msCy"></div>
            </div>
        </div>
        <div class="filter-row">
            <span class="sort-label">Metrics:</span>
            <div class="metric-checkboxes" id="metricCheckboxes"></div>
        </div>
        <div class="filter-row">
            <span class="sort-label">Sort by:</span>
            <select id="sortMetricSelect" style="font-size:11px; padding:3px 6px; border:1px solid #ccc; border-radius:4px;" onchange="changeSortMetric()"></select>
        </div>
        <div class="status-row">
            <span class="model-count" id="modelCount">0 models</span>
            <span class="selected-count" id="selectedCount">0 selected</span>
            <button id="showSelectedBtn" class="btn" onclick="toggleShowSelected()">Show Selected</button>
            <button id="airTempBtn" class="btn" onclick="toggleAirTemp()">Air Temp: OFF</button>
            <span class="loading-indicator" id="loadingIndicator">Loading...</span>
        </div>
    </div>
    <div class="explorer-banner" id="explorerBanner">
        <span id="explorerMsg"></span>
        <span class="dismiss" onclick="dismissBanner()">&times;</span>
    </div>
    <div class="table-wrap">
        <table>
            <thead><tr id="tableHeaderRow">
                <th class="cb-cell"></th>
                <th onclick="sortTable('activation')">Act</th>
                <th onclick="sortTable('num_layers')">Layers</th>
                <th onclick="sortTable('neurons')">Neurons</th>
                <th onclick="sortTable('leadtime')">LT</th>
                <th onclick="sortTable('cycle')">Cy</th>
            </tr></thead>
            <tbody id="tableBody"></tbody>
        </table>
    </div>
    <div class="resize-handle" id="resizeHandle"></div>
</div>

<div class="right-panel" id="rightPanel">
    <div class="toggle-row">
        <span style="font-size:12px;font-weight:600;color:#555;">Charts:</span>
        <button class="toggle-btn on" id="toggleTrain" onclick="toggleChart('train')">Training</button>
        <button class="toggle-btn on" id="toggleVal" onclick="toggleChart('val')">Validation</button>
        <button class="toggle-btn on" id="toggleTest" onclick="toggleChart('test')">Testing</button>
        <button class="toggle-btn on" id="toggle2021" onclick="toggleChart('2021')">2021</button>
        <button class="toggle-btn loss-btn" id="toggleLoss" onclick="toggleChart('loss')">Loss Curves</button>
        <span style="border-left:2px solid #ddd;height:20px;"></span>
        <button class="toggle-btn legend-btn on" id="toggleLegend" onclick="toggleLegends()">Legends</button>
        <button class="metric-vals-toggle on" id="toggleMetricVals" onclick="toggleMetricVals()">Metric Values</button>
    </div>
    <div class="chart-section" id="sectionTrain" style="flex:1;min-height:0;">
        <div class="chart-header">
            <span class="chart-label">Training Period</span>
        </div>
        <div class="chart-container" id="chartTrain"></div>
    </div>
    <div class="chart-section" id="sectionVal" style="flex:1;min-height:0;margin-top:4px;">
        <div class="chart-header">
            <span class="chart-label">Validation Period</span>
        </div>
        <div class="chart-container" id="chartVal"></div>
    </div>
    <div class="chart-section" id="sectionTest" style="flex:1;min-height:0;margin-top:4px;">
        <div class="chart-header">
            <span class="chart-label">Testing Period</span>
        </div>
        <div class="chart-container" id="chartTest"></div>
    </div>
    <div class="chart-section" id="section2021" style="flex:1;min-height:0;margin-top:4px;">
        <div class="chart-header">
            <span class="chart-label">2021 Independent Test</span>
        </div>
        <div class="chart-container" id="chart2021"></div>
    </div>
    <div class="chart-section" id="sectionLoss" style="flex:1;min-height:0;margin-top:4px;display:none;">
        <div class="chart-header">
            <span class="chart-label">Loss Curves (per epoch)</span>
        </div>
        <div class="chart-container" id="chartLoss"></div>
    </div>
    <div class="no-selection" id="noSelection">Check models in the table to compare time series</div>
</div>

<script>
var indexData = {index_json};
var filterValues = {filters_json};
var metricCols = {metric_cols_json};
var predUrlPrefix = '{pred_url_prefix}';
var historyUrlPrefix = '{history_url_prefix}';
var hasKerasFiles = {'true' if has_keras_files else 'false'};

var colorPalette = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
    '#a65628','#f781bf','#66c2a5','#fc8d62','#8da0cb',
    '#e78ac3','#a6d854','#ffd92f','#e5c494','#1b9e77'
];

var selectedModels = [];
var predCache = {{}};
var historyCache = {{}};
var airTempData = null;
var showAirTemp = false;
var showSelectedOnly = false;
var sortCol = metricCols[0] || 'val_mae';
var sortAsc = true;
var filteredData = [];
var pendingLoads = 0;
var showLegends = true;
var showMetricVals = true;

// Which metrics are checked (visible in table + trace labels)
var checkedMetrics = metricCols.length > 0 ? [metricCols[0]] : [];

// Chart visibility
var chartVisible = {{ train: true, val: true, test: true, '2021': true, loss: false }};

// --- Metric checkboxes ---
function initMetricCheckboxes() {{
    var container = document.getElementById('metricCheckboxes');
    container.innerHTML = '';
    metricCols.forEach(function(mc, idx) {{
        var lbl = document.createElement('label');
        lbl.className = 'metric-cb-label' + (checkedMetrics.indexOf(mc) !== -1 ? ' checked' : '');
        var cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = checkedMetrics.indexOf(mc) !== -1;
        cb.value = mc;
        cb.addEventListener('change', function() {{
            toggleMetricCheckbox(mc, this.checked);
            lbl.classList.toggle('checked', this.checked);
        }});
        lbl.appendChild(cb);
        lbl.appendChild(document.createTextNode(' ' + mc));
        container.appendChild(lbl);
    }});
    updateSortDropdown();
    updateTableHeaders();
}}

function toggleMetricCheckbox(metric, checked) {{
    if (checked) {{
        if (checkedMetrics.indexOf(metric) === -1) checkedMetrics.push(metric);
    }} else {{
        checkedMetrics = checkedMetrics.filter(function(m) {{ return m !== metric; }});
    }}
    // If sort metric was unchecked, switch to first checked
    if (checkedMetrics.indexOf(sortCol) === -1 && checkedMetrics.length > 0) {{
        sortCol = checkedMetrics[0];
    }}
    updateSortDropdown();
    updateTableHeaders();
    applyFilters();
    renderCharts();
}}

function updateSortDropdown() {{
    var sel = document.getElementById('sortMetricSelect');
    sel.innerHTML = '';
    checkedMetrics.forEach(function(mc) {{
        var opt = document.createElement('option');
        opt.value = mc;
        opt.textContent = mc;
        if (mc === sortCol) opt.selected = true;
        sel.appendChild(opt);
    }});
}}

function changeSortMetric() {{
    sortCol = document.getElementById('sortMetricSelect').value;
    sortAsc = true;
    applyFilters();
}}

function updateTableHeaders() {{
    var headerRow = document.getElementById('tableHeaderRow');
    // Remove old metric headers
    while (headerRow.children.length > 6) {{
        headerRow.removeChild(headerRow.lastChild);
    }}
    // Add checked metric headers
    checkedMetrics.forEach(function(mc) {{
        var th = document.createElement('th');
        th.textContent = mc;
        th.onclick = function() {{ sortTable(mc); }};
        headerRow.appendChild(th);
    }});
}}

// --- Resizable left panel ---
(function() {{
    var handle = document.getElementById('resizeHandle');
    var leftPanel = document.getElementById('leftPanel');
    var dragging = false;
    handle.addEventListener('mousedown', function(e) {{
        e.preventDefault();
        dragging = true;
        handle.classList.add('active');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
    }});
    document.addEventListener('mousemove', function(e) {{
        if (!dragging) return;
        var newWidth = e.clientX;
        var minW = 280;
        var maxW = window.innerWidth * 0.7;
        if (newWidth < minW) newWidth = minW;
        if (newWidth > maxW) newWidth = maxW;
        leftPanel.style.width = newWidth + 'px';
        recalcCharts();
    }});
    document.addEventListener('mouseup', function() {{
        if (dragging) {{
            dragging = false;
            handle.classList.remove('active');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            recalcCharts();
        }}
    }});
}})();

function recalcCharts() {{
    if (selectedModels.length > 0) {{
        ['chartTrain','chartVal','chartTest','chart2021','chartLoss'].forEach(function(id) {{
            var el = document.getElementById(id);
            if (el && el.style.display !== 'none') {{
                Plotly.Plots.resize(el);
            }}
        }});
    }}
}}

// --- Multi-select checkbox widget ---
var msState = {{
    activation: [],
    num_layers: [],
    neurons: [],
    leadtime: [],
    cycle: []
}};

function initMultiSelect(wrapId, filterKey) {{
    var wrap = document.getElementById(wrapId);
    var values = filterValues[filterKey];
    msState[filterKey] = values.map(function(v) {{ return String(v); }});

    var btn = document.createElement('button');
    btn.className = 'ms-btn';
    btn.textContent = 'All';
    btn.type = 'button';

    var panel = document.createElement('div');
    panel.className = 'ms-panel';

    values.forEach(function(v) {{
        var lbl = document.createElement('label');
        var cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = true;
        cb.value = String(v);
        cb.addEventListener('change', function() {{
            updateMsState(filterKey, wrapId);
        }});
        lbl.appendChild(cb);
        lbl.appendChild(document.createTextNode(' ' + v));
        panel.appendChild(lbl);
    }});

    btn.addEventListener('click', function(e) {{
        e.stopPropagation();
        document.querySelectorAll('.ms-panel.open').forEach(function(p) {{
            if (p !== panel) p.classList.remove('open');
        }});
        panel.classList.toggle('open');
    }});

    wrap.appendChild(btn);
    wrap.appendChild(panel);
}}

function updateMsState(filterKey, wrapId) {{
    var wrap = document.getElementById(wrapId);
    var cbs = wrap.querySelectorAll('input[type="checkbox"]');
    var selected = [];
    cbs.forEach(function(cb) {{
        if (cb.checked) selected.push(cb.value);
    }});
    msState[filterKey] = selected;

    var btn = wrap.querySelector('.ms-btn');
    var allVals = filterValues[filterKey];
    if (selected.length === allVals.length) {{
        btn.textContent = 'All';
    }} else if (selected.length === 0) {{
        btn.textContent = 'None';
    }} else if (selected.length <= 2) {{
        btn.textContent = selected.join(', ');
    }} else {{
        btn.textContent = selected.length + ' sel';
    }}
    applyFilters();
}}

document.addEventListener('click', function() {{
    document.querySelectorAll('.ms-panel.open').forEach(function(p) {{
        p.classList.remove('open');
    }});
}});
document.addEventListener('click', function(e) {{
    if (e.target.closest('.ms-panel')) e.stopPropagation();
}}, true);

// --- Filters ---
function applyFilters() {{
    filteredData = indexData.filter(function(m) {{
        if (msState.activation.indexOf(m.activation) === -1) return false;
        if (msState.num_layers.indexOf(String(m.num_layers)) === -1) return false;
        if (msState.neurons.indexOf(String(m.neurons)) === -1) return false;
        if (msState.leadtime.indexOf(String(m.leadtime)) === -1) return false;
        if (msState.cycle.indexOf(String(m.cycle)) === -1) return false;
        if (showSelectedOnly && selectedModels.indexOf(m.model_name) === -1) return false;
        return true;
    }});

    filteredData.sort(function(a, b) {{
        var va = a[sortCol], vb = b[sortCol];
        if (va == null) return 1;
        if (vb == null) return -1;
        if (va < vb) return sortAsc ? -1 : 1;
        if (va > vb) return sortAsc ? 1 : -1;
        return 0;
    }});

    renderTable();
}}

function sortTable(col) {{
    if (sortCol === col) {{ sortAsc = !sortAsc; }}
    else {{ sortCol = col; sortAsc = true; }}
    applyFilters();
}}

function toggleShowSelected() {{
    showSelectedOnly = !showSelectedOnly;
    var btn = document.getElementById('showSelectedBtn');
    btn.textContent = showSelectedOnly ? 'Show All' : 'Show Selected';
    btn.classList.toggle('active', showSelectedOnly);
    applyFilters();
}}

function toggleAirTemp() {{
    showAirTemp = !showAirTemp;
    var btn = document.getElementById('airTempBtn');
    btn.textContent = showAirTemp ? 'Air Temp: ON' : 'Air Temp: OFF';
    btn.classList.toggle('active', showAirTemp);

    if (showAirTemp && !airTempData) {{
        loadAirTemp(function() {{ renderCharts(); }});
    }} else {{
        renderCharts();
    }}
}}

// --- Chart toggle buttons ---
function toggleChart(which) {{
    chartVisible[which] = !chartVisible[which];
    var btnMap = {{
        'train': 'toggleTrain', 'val': 'toggleVal', 'test': 'toggleTest',
        '2021': 'toggle2021', 'loss': 'toggleLoss'
    }};
    var btn = document.getElementById(btnMap[which]);
    btn.classList.toggle('on', chartVisible[which]);

    // When loss is toggled on for the first time, load histories for selected models
    if (which === 'loss' && chartVisible.loss) {{
        var needLoad = [];
        selectedModels.forEach(function(name) {{
            if (!historyCache[name] && historyCache[name] !== null) {{
                needLoad.push(name);
            }}
        }});
        if (needLoad.length > 0) {{
            var loaded = 0;
            needLoad.forEach(function(name) {{
                loadHistory(name, function() {{
                    loaded++;
                    if (loaded === needLoad.length) renderCharts();
                }});
            }});
            return;
        }}
    }}
    renderCharts();
}}

function toggleLegends() {{
    showLegends = !showLegends;
    document.getElementById('toggleLegend').classList.toggle('on', showLegends);
    renderCharts();
}}

function toggleMetricVals() {{
    showMetricVals = !showMetricVals;
    document.getElementById('toggleMetricVals').classList.toggle('on', showMetricVals);
    renderCharts();
}}

function getModelColor(name) {{
    var idx = selectedModels.indexOf(name);
    return idx === -1 ? null : colorPalette[idx % colorPalette.length];
}}

// --- Build metric string for a model's trace label ---
function buildMetricLabel(info) {{
    if (!showMetricVals || checkedMetrics.length === 0) return '';
    var parts = [];
    checkedMetrics.forEach(function(mc) {{
        var val = info[mc];
        if (val != null) parts.push(mc + '=' + val.toFixed(4));
    }});
    return parts.length > 0 ? ' [' + parts.join(', ') + ']' : '';
}}

// --- Build short model label ---
function buildModelLabel(info) {{
    return info.activation + ' ' + info.num_layers + 'L ' + info.neurons + 'N lt' + info.leadtime + ' c' + info.cycle;
}}

function renderTable() {{
    var tbody = document.getElementById('tableBody');
    tbody.innerHTML = '';
    filteredData.forEach(function(m) {{
        var tr = document.createElement('tr');
        var isSelected = selectedModels.indexOf(m.model_name) !== -1;
        if (isSelected) tr.className = 'selected';
        var color = getModelColor(m.model_name);
        var dotHtml = color ? '<span class="color-dot" style="background:' + color + '"></span>' : '';

        var cells =
            '<td class="cb-cell">' + dotHtml +
            '<input type="checkbox" ' + (isSelected ? 'checked' : '') +
            ' onchange="toggleModel(\\'' + m.model_name.replace(/'/g, "\\\\'") + '\\', this.checked)"></td>' +
            '<td>' + m.activation + '</td>' +
            '<td>' + m.num_layers + '</td>' +
            '<td>' + m.neurons + '</td>' +
            '<td>' + m.leadtime + 'h</td>' +
            '<td>' + m.cycle + '</td>';

        // Add cells for each checked metric
        checkedMetrics.forEach(function(mc) {{
            var val = m[mc];
            cells += '<td>' + (val != null ? val.toFixed(4) : '-') + '</td>';
        }});

        tr.innerHTML = cells;
        tbody.appendChild(tr);
    }});
    document.getElementById('modelCount').textContent = filteredData.length + ' models';
    document.getElementById('selectedCount').textContent = selectedModels.length + ' selected';
}}

// --- CSV parsing ---
function parseCSV(text) {{
    var lines = text.trim().split('\\n');
    var headers = lines[0].split(',');
    var rows = [];
    for (var i = 1; i < lines.length; i++) {{
        var vals = lines[i].split(',');
        var row = {{}};
        headers.forEach(function(h, j) {{ row[h.trim()] = vals[j] ? vals[j].trim() : ''; }});
        rows.push(row);
    }}
    return rows;
}}

function processPredCSV(rows) {{
    var train = {{ dates: [], actual: [], predicted: [] }};
    var test = {{ dates: [], actual: [], predicted: [] }};
    var val = {{ dates: [], actual: [], predicted: [] }};
    var y2021 = {{ dates: [], actual: [], predicted: [] }};
    rows.forEach(function(r) {{
        var target;
        if (r.dataset === 'train') target = train;
        else if (r.dataset === 'test') target = test;
        else if (r.dataset === 'val') target = val;
        else target = y2021;  // '2021'
        target.dates.push(r.date);
        target.actual.push(parseFloat(r.actual));
        target.predicted.push(parseFloat(r.predicted));
    }});
    return {{ train: train, test: test, val: val, '2021': y2021 }};
}}

// --- Load air temp CSV ---
function loadAirTemp(callback) {{
    if (airTempData) {{ callback(); return; }}
    var url = predUrlPrefix + '/air_temp.csv';
    fetch(url)
        .then(function(resp) {{
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            return resp.text();
        }})
        .then(function(text) {{
            var rows = parseCSV(text);
            airTempData = {{ dates: [], temps: [] }};
            rows.forEach(function(r) {{
                airTempData.dates.push(r.date);
                airTempData.temps.push(parseFloat(r.air_temp));
            }});
            callback();
        }})
        .catch(function(err) {{
            console.error('Failed to load air_temp.csv:', err);
            airTempData = {{ dates: [], temps: [] }};
            callback();
        }});
}}

// --- Load prediction CSV on demand ---
function loadPrediction(name, callback) {{
    if (predCache[name]) {{
        callback(predCache[name]);
        return;
    }}
    pendingLoads++;
    document.getElementById('loadingIndicator').style.display = '';

    var url = predUrlPrefix + '/' + name + '_pred.csv';
    fetch(url)
        .then(function(resp) {{
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            return resp.text();
        }})
        .then(function(text) {{
            var data = processPredCSV(parseCSV(text));
            predCache[name] = data;
            callback(data);
        }})
        .catch(function(err) {{
            console.error('Failed to load ' + name + ':', err);
            callback(null);
        }})
        .finally(function() {{
            pendingLoads--;
            if (pendingLoads <= 0) {{
                pendingLoads = 0;
                document.getElementById('loadingIndicator').style.display = 'none';
            }}
        }});
}}

// --- Load training history JSON on demand ---
function loadHistory(name, callback) {{
    if (historyCache.hasOwnProperty(name)) {{
        callback(historyCache[name]);
        return;
    }}
    pendingLoads++;
    document.getElementById('loadingIndicator').style.display = '';

    var url = historyUrlPrefix + '/' + name + '_history.json';
    fetch(url)
        .then(function(resp) {{
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            return resp.json();
        }})
        .then(function(data) {{
            historyCache[name] = data;
            callback(data);
        }})
        .catch(function(err) {{
            console.error('Failed to load history for ' + name + ':', err);
            historyCache[name] = null;
            callback(null);
        }})
        .finally(function() {{
            pendingLoads--;
            if (pendingLoads <= 0) {{
                pendingLoads = 0;
                document.getElementById('loadingIndicator').style.display = 'none';
            }}
        }});
}}

// --- Toggle model ---
function toggleModel(name, checked) {{
    if (checked) {{
        if (selectedModels.indexOf(name) === -1) selectedModels.push(name);
        // Load prediction
        loadPrediction(name, function() {{
            // Also load history if loss chart is visible
            if (chartVisible.loss) {{
                loadHistory(name, function() {{
                    renderTable();
                    renderCharts();
                }});
            }} else {{
                renderTable();
                renderCharts();
            }}
        }});
    }} else {{
        selectedModels = selectedModels.filter(function(n) {{ return n !== name; }});
        renderTable();
        renderCharts();
    }}
    renderTable();
}}

// --- Build air temp trace filtered to a date range ---
function buildAirTempTrace(dates, legendName, showInLegend) {{
    if (!airTempData || airTempData.dates.length === 0) return null;
    var minDate = dates[0];
    var maxDate = dates[dates.length - 1];
    var filtDates = [];
    var filtTemps = [];
    for (var i = 0; i < airTempData.dates.length; i++) {{
        var d = airTempData.dates[i];
        if (d >= minDate && d <= maxDate) {{
            filtDates.push(d);
            filtTemps.push(airTempData.temps[i]);
        }}
    }}
    if (filtDates.length === 0) return null;
    return {{
        x: filtDates, y: filtTemps,
        mode: 'lines', name: legendName,
        line: {{ color: '#ff9999', width: 1, dash: 'dot' }},
        legendgroup: 'air_temp', showlegend: showInLegend,
        visible: true
    }};
}}

// --- Render charts ---
function renderCharts() {{
    var noSel = document.getElementById('noSelection');
    var sectionTrain = document.getElementById('sectionTrain');
    var sectionVal = document.getElementById('sectionVal');
    var sectionTest = document.getElementById('sectionTest');
    var section2021 = document.getElementById('section2021');
    var sectionLoss = document.getElementById('sectionLoss');
    var chartTrain = document.getElementById('chartTrain');
    var chartVal = document.getElementById('chartVal');
    var chartTest = document.getElementById('chartTest');
    var chart2021 = document.getElementById('chart2021');
    var chartLoss = document.getElementById('chartLoss');

    if (selectedModels.length === 0) {{
        noSel.style.display = 'flex';
        sectionTrain.style.display = 'none';
        sectionVal.style.display = 'none';
        sectionTest.style.display = 'none';
        section2021.style.display = 'none';
        sectionLoss.style.display = 'none';
        return;
    }}
    noSel.style.display = 'none';

    // Show/hide chart sections based on toggle
    sectionTrain.style.display = chartVisible.train ? '' : 'none';
    sectionVal.style.display = chartVisible.val ? '' : 'none';
    sectionTest.style.display = chartVisible.test ? '' : 'none';
    section2021.style.display = chartVisible['2021'] ? '' : 'none';
    sectionLoss.style.display = chartVisible.loss ? '' : 'none';

    var trainTraces = [];
    var valTraces = [];
    var testTraces = [];
    var y2021Traces = [];
    var lossTraces = [];
    var airShownTrain = false;
    var airShownVal = false;
    var airShownTest = false;
    var airShown2021 = false;

    selectedModels.forEach(function(name, idx) {{
        var pred = predCache[name];
        if (!pred) return;
        var color = colorPalette[idx % colorPalette.length];
        var info = indexData.find(function(m) {{ return m.model_name === name; }});
        var baseLabel = info ? buildModelLabel(info) : name;
        var metricSuffix = info ? buildMetricLabel(info) : '';
        var label = baseLabel + metricSuffix;

        // --- Training chart ---
        if (chartVisible.train && pred.train && pred.train.dates.length > 0) {{
            if (idx === 0) {{
                trainTraces.push({{
                    x: pred.train.dates, y: pred.train.actual,
                    mode: 'lines', name: 'Observed (Train)',
                    line: {{ color: 'black', width: 1.5 }},
                    legendgroup: 'actual_train', showlegend: true
                }});
            }}
            trainTraces.push({{
                x: pred.train.dates, y: pred.train.predicted,
                mode: 'lines', name: label,
                line: {{ color: color, width: 1.2 }},
                legendgroup: name + '_train'
            }});
            if (showAirTemp && airTempData && !airShownTrain) {{
                var atTrain = buildAirTempTrace(pred.train.dates, 'Air Temp', true);
                if (atTrain) {{ trainTraces.push(atTrain); airShownTrain = true; }}
            }}
        }}

        // --- Validation chart ---
        if (chartVisible.val && pred.val.dates.length > 0) {{
            if (idx === 0) {{
                valTraces.push({{
                    x: pred.val.dates, y: pred.val.actual,
                    mode: 'lines', name: 'Observed (Val)',
                    line: {{ color: 'black', width: 1.5 }},
                    legendgroup: 'actual_val', showlegend: true
                }});
            }}
            valTraces.push({{
                x: pred.val.dates, y: pred.val.predicted,
                mode: 'lines', name: label,
                line: {{ color: color, width: 1.2 }},
                legendgroup: name + '_val'
            }});
            if (showAirTemp && airTempData && !airShownVal) {{
                var atVal = buildAirTempTrace(pred.val.dates, 'Air Temp', true);
                if (atVal) {{ valTraces.push(atVal); airShownVal = true; }}
            }}
        }}

        // --- Testing chart ---
        if (chartVisible.test && pred.test && pred.test.dates.length > 0) {{
            if (idx === 0) {{
                testTraces.push({{
                    x: pred.test.dates, y: pred.test.actual,
                    mode: 'lines', name: 'Observed (Test)',
                    line: {{ color: 'black', width: 1.5 }},
                    legendgroup: 'actual_test', showlegend: true
                }});
            }}
            testTraces.push({{
                x: pred.test.dates, y: pred.test.predicted,
                mode: 'lines', name: label,
                line: {{ color: color, width: 1.2 }},
                legendgroup: name + '_test'
            }});
            if (showAirTemp && airTempData && !airShownTest) {{
                var atTest = buildAirTempTrace(pred.test.dates, 'Air Temp', true);
                if (atTest) {{ testTraces.push(atTest); airShownTest = true; }}
            }}
        }}

        // --- 2021 chart ---
        if (chartVisible['2021'] && pred['2021'].dates.length > 0) {{
            if (idx === 0) {{
                y2021Traces.push({{
                    x: pred['2021'].dates, y: pred['2021'].actual,
                    mode: 'lines', name: 'Observed (2021)',
                    line: {{ color: 'black', width: 1.5 }},
                    legendgroup: 'actual_2021', showlegend: true
                }});
            }}
            y2021Traces.push({{
                x: pred['2021'].dates, y: pred['2021'].predicted,
                mode: 'lines', name: label,
                line: {{ color: color, width: 1.2 }},
                legendgroup: name + '_2021'
            }});
            if (showAirTemp && airTempData && !airShown2021) {{
                var at2021 = buildAirTempTrace(pred['2021'].dates, 'Air Temp', true);
                if (at2021) {{ y2021Traces.push(at2021); airShown2021 = true; }}
            }}
        }}

        // --- Loss curves chart ---
        if (chartVisible.loss) {{
            var hist = historyCache[name];
            if (hist && hist.loss) {{
                var epochs = hist.loss.map(function(_, i) {{ return i + 1; }});
                // Training loss (solid line)
                lossTraces.push({{
                    x: epochs, y: hist.loss,
                    mode: 'lines', name: baseLabel + ' (loss)',
                    line: {{ color: color, width: 1.5 }},
                    legendgroup: name + '_loss'
                }});
                // Validation loss (dashed line, same color)
                if (hist.val_loss) {{
                    lossTraces.push({{
                        x: epochs, y: hist.val_loss,
                        mode: 'lines', name: baseLabel + ' (val_loss)',
                        line: {{ color: color, width: 1.5, dash: 'dash' }},
                        legendgroup: name + '_loss'
                    }});
                }}
            }}
        }}
    }});

    // Count visible sections for height calculation
    var visibleCount = 0;
    if (chartVisible.train) visibleCount++;
    if (chartVisible.val) visibleCount++;
    if (chartVisible.test) visibleCount++;
    if (chartVisible['2021']) visibleCount++;
    if (chartVisible.loss) visibleCount++;
    if (visibleCount === 0) visibleCount = 1;
    var h = (window.innerHeight - 80) / visibleCount;

    var layoutBase = {{
        margin: {{ t: 10, b: 40, l: 50, r: 20 }},
        hovermode: 'x unified',
        legend: {{ orientation: 'h', y: 1.02, yanchor: 'bottom', font: {{ size: 10 }} }},
        yaxis: {{ title: 'Temperature (\\u00b0C)' }},
        xaxis: {{ type: 'date' }},
        showlegend: showLegends,
    }};

    if (chartVisible.train && trainTraces.length > 0) {{
        chartTrain.style.display = 'block';
        Plotly.newPlot('chartTrain', trainTraces,
            Object.assign({{}}, layoutBase, {{ height: h }}),
            {{ responsive: true }});
    }} else {{
        chartTrain.style.display = 'none';
    }}

    if (chartVisible.val && valTraces.length > 0) {{
        chartVal.style.display = 'block';
        Plotly.newPlot('chartVal', valTraces,
            Object.assign({{}}, layoutBase, {{ height: h }}),
            {{ responsive: true }});
    }} else {{
        chartVal.style.display = 'none';
    }}

    if (chartVisible.test && testTraces.length > 0) {{
        chartTest.style.display = 'block';
        Plotly.newPlot('chartTest', testTraces,
            Object.assign({{}}, layoutBase, {{ height: h }}),
            {{ responsive: true }});
    }} else {{
        chartTest.style.display = 'none';
    }}

    if (chartVisible['2021'] && y2021Traces.length > 0) {{
        chart2021.style.display = 'block';
        Plotly.newPlot('chart2021', y2021Traces,
            Object.assign({{}}, layoutBase, {{ height: h }}),
            {{ responsive: true }});
    }} else {{
        chart2021.style.display = 'none';
    }}

    // Loss curves chart uses different axes
    if (chartVisible.loss && lossTraces.length > 0) {{
        chartLoss.style.display = 'block';
        var lossLayout = {{
            margin: {{ t: 10, b: 40, l: 50, r: 20 }},
            hovermode: 'x unified',
            legend: {{ orientation: 'h', y: 1.02, yanchor: 'bottom', font: {{ size: 10 }} }},
            yaxis: {{ title: 'Loss' }},
            xaxis: {{ title: 'Epoch' }},
            showlegend: showLegends,
            height: h
        }};
        Plotly.newPlot('chartLoss', lossTraces, lossLayout, {{ responsive: true }});
    }} else if (chartVisible.loss) {{
        chartLoss.style.display = 'block';
        chartLoss.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#aaa;font-size:14px;">No history data found for selected models</div>';
    }} else {{
        chartLoss.style.display = 'none';
    }}
}}

// --- Explorer integration via localStorage ---
function loadFromLocalStorage() {{
    var raw = localStorage.getItem('tsModels');
    if (!raw) return;

    var models;
    try {{ models = JSON.parse(raw); }} catch(e) {{ return; }}
    if (!Array.isArray(models) || models.length === 0) return;

    var matched = [];
    models.forEach(function(criteria) {{
        indexData.forEach(function(m) {{
            if (m.activation === criteria.activation &&
                m.num_layers === criteria.num_layers &&
                m.neurons === criteria.neurons &&
                m.leadtime === criteria.leadtime &&
                m.cycle === criteria.cycle &&
                matched.indexOf(m.model_name) === -1) {{
                matched.push(m.model_name);
            }}
        }});
    }});

    if (matched.length === 0) return;

    matched.forEach(function(name) {{
        if (selectedModels.indexOf(name) === -1) selectedModels.push(name);
    }});

    var banner = document.getElementById('explorerBanner');
    document.getElementById('explorerMsg').textContent =
        matched.length + ' model(s) loaded from Explorer';
    banner.classList.add('visible');

    var loadCount = 0;
    matched.forEach(function(name) {{
        loadPrediction(name, function() {{
            loadCount++;
            if (loadCount === matched.length) {{
                applyFilters();
                renderCharts();
            }}
        }});
    }});
    applyFilters();
}}

function refreshFromExplorer() {{
    loadFromLocalStorage();
}}

function dismissBanner() {{
    document.getElementById('explorerBanner').classList.remove('visible');
    localStorage.removeItem('tsModels');
    localStorage.removeItem('tsTimestamp');
}}

// --- Window resize ---
window.addEventListener('resize', function() {{
    recalcCharts();
}});

// --- Init ---
initMultiSelect('msAct', 'activation');
initMultiSelect('msLayers', 'num_layers');
initMultiSelect('msNeurons', 'neurons');
initMultiSelect('msLT', 'leadtime');
initMultiSelect('msCy', 'cycle');

initMetricCheckboxes();
applyFilters();
loadFromLocalStorage();
</script>
</body></html>"""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_file = f"{output_dir}/05_timeseries.html"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved: 05_timeseries.html ({len(index_records)} models in picker)")
    return html
