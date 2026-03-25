"""
V14 Time Series Comparison: model picker + dual stacked Plotly charts.

Changes from v13:
- Reads localStorage for models sent from the Explorer (plotly_selected)
- Refresh button to force reload from localStorage
- No zoom sync between val and 2021 charts (different date ranges)
- Always draws the observation (actual) line per model, even if duplicated

Usage: run via serve_timeseries.py which starts a local HTTP server and
opens the HTML in a browser.
"""

from typing import Optional
from pathlib import Path
import json as _json
import os

import pandas as pd


def plot_timeseries_compare_v14(predictions_dir: str, output_dir: str) -> Optional[str]:
    """Generate the time series comparison HTML.

    Args:
        predictions_dir: path to predictions/ folder with _index.csv and *_pred.csv
        output_dir: where to write the HTML
    """
    pred_dir = Path(predictions_dir)
    index_path = pred_dir / '_index.csv'
    if not index_path.exists():
        print("No _index.csv found in predictions/. Run generate_predictions.py first.")
        return None

    index_df = pd.read_csv(index_path)
    if len(index_df) == 0:
        print("_index.csv is empty.")
        return None

    # Build index records for JS
    index_records = []
    for _, row in index_df.iterrows():
        index_records.append({
            'model_name': str(row['model_name']),
            'activation': str(row.get('activation', '')),
            'num_layers': int(row.get('num_layers', 0)),
            'neurons': int(row.get('neurons', 0)),
            'leadtime': int(row.get('leadtime', 0)),
            'cycle': int(row.get('cycle', 0)),
            'val_mae': round(float(row['val_mae']), 4) if pd.notna(row.get('val_mae')) else None,
            'mae_2021': round(float(row['mae_2021']), 4) if pd.notna(row.get('mae_2021')) else None,
        })

    filter_values = {
        'activation': sorted(index_df['activation'].dropna().unique().tolist()),
        'num_layers': sorted(index_df['num_layers'].dropna().unique().astype(int).tolist()),
        'neurons': sorted(index_df['neurons'].dropna().unique().astype(int).tolist()),
        'leadtime': sorted(index_df['leadtime'].dropna().unique().astype(int).tolist()),
        'cycle': sorted(index_df['cycle'].dropna().unique().astype(int).tolist()),
    }

    # Compute relative path from output_dir to predictions_dir
    out_path = Path(output_dir).resolve()
    pred_path = pred_dir.resolve()
    try:
        rel_pred = os.path.relpath(pred_path, out_path)
        pred_url_prefix = rel_pred.replace('\\', '/')
    except ValueError:
        pred_url_prefix = str(pred_path).replace('\\', '/')

    index_json = _json.dumps(index_records)
    filters_json = _json.dumps(filter_values)

    html = f"""<!DOCTYPE html>
<html><head>
<title>Time Series Comparison</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           display: flex; height: 100vh; overflow: hidden; }}
    .left-panel {{ width: 35%; min-width: 380px; border-right: 2px solid #ddd;
                   display: flex; flex-direction: column; background: #fafafa; }}
    .panel-header {{ padding: 12px 16px; background: #f0f0f0; border-bottom: 1px solid #ddd; }}
    .panel-header h2 {{ font-size: 16px; margin-bottom: 8px; }}
    .filters {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }}
    .filters label {{ font-size: 12px; font-weight: 600; color: #555; }}
    .filters select {{ font-size: 12px; padding: 2px 6px; border: 1px solid #ccc; border-radius: 3px; }}
    .model-count {{ font-size: 12px; color: #888; }}
    .selected-count {{ font-size: 12px; font-weight: 600; color: #52a8ad; }}
    .loading-indicator {{ font-size: 11px; color: #ff7f00; font-weight: 600; display: none; }}
    .explorer-banner {{ font-size: 12px; padding: 6px 12px; background: #fff3e0; border-bottom: 1px solid #ffcc80;
                        display: none; align-items: center; gap: 8px; }}
    .explorer-banner.visible {{ display: flex; }}
    .explorer-banner .dismiss {{ cursor: pointer; color: #999; font-weight: bold; margin-left: auto; }}
    .explorer-banner .dismiss:hover {{ color: #333; }}
    .btn-refresh {{ font-size: 11px; padding: 3px 10px; border: 1px solid #52a8ad; border-radius: 4px;
                    background: white; cursor: pointer; color: #52a8ad; font-weight: 600; }}
    .btn-refresh:hover {{ background: #52a8ad; color: white; }}
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
    .right-panel {{ flex: 1; display: flex; flex-direction: column; padding: 8px; }}
    .chart-container {{ flex: 1; min-height: 0; }}
    .chart-label {{ font-size: 13px; font-weight: 600; color: #555; padding: 4px 8px; }}
    .no-selection {{ display: flex; align-items: center; justify-content: center;
                     height: 100%; color: #aaa; font-size: 16px; }}
</style>
</head><body>

<div class="left-panel">
    <div class="panel-header">
        <h2>Model Picker
            <button class="btn-refresh" onclick="refreshFromExplorer()" title="Reload models sent from Explorer">Refresh</button>
        </h2>
        <div class="filters">
            <div><label>Activation</label><br><select id="fAct" onchange="applyFilters()"><option value="all">All</option></select></div>
            <div><label>Layers</label><br><select id="fLayers" onchange="applyFilters()"><option value="all">All</option></select></div>
            <div><label>Neurons</label><br><select id="fNeurons" onchange="applyFilters()"><option value="all">All</option></select></div>
            <div><label>Leadtime</label><br><select id="fLT" onchange="applyFilters()"><option value="all">All</option></select></div>
            <div><label>Cycle</label><br><select id="fCycle" onchange="applyFilters()"><option value="all">All</option></select></div>
        </div>
        <span class="model-count" id="modelCount">0 models</span>
        &nbsp;|&nbsp;
        <span class="selected-count" id="selectedCount">0 selected</span>
        &nbsp;
        <span class="loading-indicator" id="loadingIndicator">Loading...</span>
    </div>
    <div class="explorer-banner" id="explorerBanner">
        <span id="explorerMsg"></span>
        <span class="dismiss" onclick="dismissBanner()">&times;</span>
    </div>
    <div class="table-wrap">
        <table>
            <thead><tr>
                <th class="cb-cell"></th>
                <th onclick="sortTable('activation')">Act</th>
                <th onclick="sortTable('num_layers')">Layers</th>
                <th onclick="sortTable('neurons')">Neurons</th>
                <th onclick="sortTable('leadtime')">LT</th>
                <th onclick="sortTable('cycle')">Cy</th>
                <th onclick="sortTable('val_mae')">val_mae</th>
                <th onclick="sortTable('mae_2021')">mae_2021</th>
            </tr></thead>
            <tbody id="tableBody"></tbody>
        </table>
    </div>
</div>

<div class="right-panel">
    <div class="chart-label">Validation Period</div>
    <div class="chart-container" id="chartVal"></div>
    <div class="chart-label" style="margin-top:4px;">2021 Independent Test</div>
    <div class="chart-container" id="chart2021"></div>
    <div class="no-selection" id="noSelection">Check models in the table to compare time series</div>
</div>

<script>
var indexData = {index_json};
var filterValues = {filters_json};
var predUrlPrefix = '{pred_url_prefix}';

var colorPalette = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
    '#a65628','#f781bf','#66c2a5','#fc8d62','#8da0cb',
    '#e78ac3','#a6d854','#ffd92f','#e5c494','#1b9e77'
];

var selectedModels = [];
var predCache = {{}};
var sortCol = 'val_mae';
var sortAsc = true;
var filteredData = [];
var pendingLoads = 0;

function initFilters() {{
    var map = {{
        'activation': 'fAct', 'num_layers': 'fLayers',
        'neurons': 'fNeurons', 'leadtime': 'fLT', 'cycle': 'fCycle'
    }};
    Object.keys(map).forEach(function(key) {{
        var sel = document.getElementById(map[key]);
        filterValues[key].forEach(function(v) {{
            var opt = document.createElement('option');
            opt.value = String(v);
            opt.textContent = String(v);
            sel.appendChild(opt);
        }});
    }});
}}

function applyFilters() {{
    var fAct = document.getElementById('fAct').value;
    var fLayers = document.getElementById('fLayers').value;
    var fNeurons = document.getElementById('fNeurons').value;
    var fLT = document.getElementById('fLT').value;
    var fCycle = document.getElementById('fCycle').value;

    filteredData = indexData.filter(function(m) {{
        if (fAct !== 'all' && m.activation !== fAct) return false;
        if (fLayers !== 'all' && String(m.num_layers) !== fLayers) return false;
        if (fNeurons !== 'all' && String(m.neurons) !== fNeurons) return false;
        if (fLT !== 'all' && String(m.leadtime) !== fLT) return false;
        if (fCycle !== 'all' && String(m.cycle) !== fCycle) return false;
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

function getModelColor(name) {{
    var idx = selectedModels.indexOf(name);
    return idx === -1 ? null : colorPalette[idx % colorPalette.length];
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
        tr.innerHTML =
            '<td class="cb-cell">' + dotHtml +
            '<input type="checkbox" ' + (isSelected ? 'checked' : '') +
            ' onchange="toggleModel(\\'' + m.model_name.replace(/'/g, "\\\\'") + '\\', this.checked)"></td>' +
            '<td>' + m.activation + '</td>' +
            '<td>' + m.num_layers + '</td>' +
            '<td>' + m.neurons + '</td>' +
            '<td>' + m.leadtime + 'h</td>' +
            '<td>' + m.cycle + '</td>' +
            '<td>' + (m.val_mae != null ? m.val_mae.toFixed(4) : '-') + '</td>' +
            '<td>' + (m.mae_2021 != null ? m.mae_2021.toFixed(4) : '-') + '</td>';
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
    var val = {{ dates: [], actual: [], predicted: [] }};
    var test = {{ dates: [], actual: [], predicted: [] }};
    rows.forEach(function(r) {{
        var target = r.dataset === 'val' ? val : test;
        target.dates.push(r.date);
        target.actual.push(parseFloat(r.actual));
        target.predicted.push(parseFloat(r.predicted));
    }});
    return {{ val: val, '2021': test }};
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

// --- Toggle model ---
function toggleModel(name, checked) {{
    if (checked) {{
        if (selectedModels.indexOf(name) === -1) selectedModels.push(name);
        loadPrediction(name, function() {{ renderTable(); renderCharts(); }});
    }} else {{
        selectedModels = selectedModels.filter(function(n) {{ return n !== name; }});
        renderTable();
        renderCharts();
    }}
    renderTable();
}}

// --- Render charts ---
function renderCharts() {{
    var noSel = document.getElementById('noSelection');
    var chartVal = document.getElementById('chartVal');
    var chart2021 = document.getElementById('chart2021');

    if (selectedModels.length === 0) {{
        noSel.style.display = 'flex';
        chartVal.style.display = 'none';
        chart2021.style.display = 'none';
        return;
    }}
    noSel.style.display = 'none';
    chartVal.style.display = 'block';
    chart2021.style.display = 'block';

    var valTraces = [];
    var testTraces = [];

    selectedModels.forEach(function(name, idx) {{
        var pred = predCache[name];
        if (!pred) return;
        var color = colorPalette[idx % colorPalette.length];
        var info = indexData.find(function(m) {{ return m.model_name === name; }});
        var label = info ? info.activation + ' ' + info.num_layers + 'L ' + info.neurons + 'N lt' + info.leadtime + ' c' + info.cycle : name;

        // Always draw observation line for each model (even if duplicated)
        if (pred.val.dates.length > 0) {{
            valTraces.push({{
                x: pred.val.dates, y: pred.val.actual,
                mode: 'lines', name: 'Actual (val)',
                line: {{ color: 'black', width: 1.5 }},
                legendgroup: 'actual_val', showlegend: (idx === 0)
            }});
        }}
        valTraces.push({{
            x: pred.val.dates, y: pred.val.predicted,
            mode: 'lines', name: label,
            line: {{ color: color, width: 1.2 }},
            legendgroup: name
        }});

        if (pred['2021'].dates.length > 0) {{
            testTraces.push({{
                x: pred['2021'].dates, y: pred['2021'].actual,
                mode: 'lines', name: 'Actual (2021)',
                line: {{ color: 'black', width: 1.5 }},
                legendgroup: 'actual_2021', showlegend: (idx === 0)
            }});
        }}
        testTraces.push({{
            x: pred['2021'].dates, y: pred['2021'].predicted,
            mode: 'lines', name: label,
            line: {{ color: color, width: 1.2 }},
            legendgroup: name
        }});
    }});

    var layoutBase = {{
        margin: {{ t: 10, b: 40, l: 50, r: 20 }},
        hovermode: 'x unified',
        legend: {{ orientation: 'h', y: 1.02, yanchor: 'bottom', font: {{ size: 11 }} }},
        yaxis: {{ title: 'Water Temp (\\u00b0C)' }},
        xaxis: {{ type: 'date' }},
    }};

    var h = (window.innerHeight - 80) / 2;

    Plotly.newPlot('chartVal', valTraces,
        Object.assign({{}}, layoutBase, {{ height: h }}),
        {{ responsive: true }});

    Plotly.newPlot('chart2021', testTraces,
        Object.assign({{}}, layoutBase, {{ height: h }}),
        {{ responsive: true }});

    // No zoom sync — charts cover different date ranges
}}

// --- Explorer integration via localStorage ---
function loadFromLocalStorage() {{
    var raw = localStorage.getItem('tsModels');
    if (!raw) return;

    var models;
    try {{ models = JSON.parse(raw); }} catch(e) {{ return; }}
    if (!Array.isArray(models) || models.length === 0) return;

    // Match explorer criteria to index model names
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

    // Auto-select matched models
    matched.forEach(function(name) {{
        if (selectedModels.indexOf(name) === -1) selectedModels.push(name);
    }});

    // Show banner
    var banner = document.getElementById('explorerBanner');
    document.getElementById('explorerMsg').textContent =
        matched.length + ' model(s) loaded from Explorer';
    banner.classList.add('visible');

    // Load predictions and render
    var loadCount = 0;
    matched.forEach(function(name) {{
        loadPrediction(name, function() {{
            loadCount++;
            if (loadCount === matched.length) {{
                renderTable();
                renderCharts();
            }}
        }});
    }});
    renderTable();
}}

function refreshFromExplorer() {{
    loadFromLocalStorage();
}}

function dismissBanner() {{
    document.getElementById('explorerBanner').classList.remove('visible');
    localStorage.removeItem('tsModels');
    localStorage.removeItem('tsTimestamp');
}}

// --- Init ---
initFilters();
applyFilters();
loadFromLocalStorage();
</script>
</body></html>"""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{output_dir}/05_timeseries.html"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved: 05_timeseries.html ({len(index_records)} models in picker)")
    return html
