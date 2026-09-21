"""
V17 Unified Plot: Scatter / Boxplot / Violin in one HTML.

Same as v15, plus ON-THE-FLY aggregation of repetitions:
- Show mode toggle: Raw | Avg | Avg+Spread
    * Raw        -> every repetition is its own dot (unchanged v15 behavior)
    * Avg        -> collapse the reps of each combo to one mean dot
    * Avg+Spread -> mean dot + error fence (scatter only)
- Spread dropdown (Avg+Spread only): Std Dev | 5-95 | 0-100 (min-max) | 10-90
- All computed in the browser from the raw per-rep rows, so ONE file/dataset
  (mape_progress_reps1to10.csv) serves both the raw cloud and the summary view.

Guardrails (loud, not silent):
1. If the data has <2 reps per combo (i.e. an already-averaged file), the Avg /
   Avg+Spread buttons are disabled with an explanatory tooltip and the plot stays
   in Raw mode.
2. In Avg / Avg+Spread mode the hover reads "mean of N reps" (N is the actual
   rep count for that combo, in case it is <10), and Avg+Spread also shows the
   lo-hi fence bounds.

Combo identity = (activation, num_layers, neurons, leadtime, cycle) -- reps of one
combo share identical hyperparameters, so that tuple is the grouping key.
"""

from typing import Optional
import json as _json
import pandas as pd

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_unified_v18(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[str]:
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()

    has_dropout = 'dropout' in clean.columns

    # dropout is a grid dimension, not a metric — skip it so it doesn't show up in
    # the metric/Y dropdowns, and (critically) so it becomes part of the combo key.
    _skip = {'run_id', 'iteration', 'leadtime', 'cycle', 'num_layers', 'neurons', 'dropout'}
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
        # The rep number. data_loader renames the CSV's run_num -> run_id, but
        # the page hovers and exports it as run_num (the name the tuner and the
        # ops selection sheet both use), so translate back at this boundary
        # rather than leaking the internal name. Without this the field is
        # simply absent, which is why export wrote run_num: null.
        if 'run_id' in clean.columns:
            try:
                rec['run_num'] = int(row['run_id'])
            except (ValueError, TypeError):
                rec['run_num'] = None
        if has_dropout:
            try:
                rec['dropout'] = float(row['dropout'])
            except (ValueError, TypeError):
                rec['dropout'] = 0.0
        # Compact per-combo label so 'combo' can be used as an X-axis category
        # (one box/dot per config). dropout included so variants stay distinct.
        rec['combo'] = (f"LT{rec['leadtime']} c{rec['cycle']} {rec['activation']} "
                        f"{rec['num_layers']}L {rec['neurons']}N"
                        + (f" d{rec['dropout']}" if has_dropout else ""))
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

    _fv = {
        'activation': activations,
        'num_layers': num_layers_vals,
        'neurons': neurons_vals,
        'leadtime': leadtime_vals,
        'cycle': cycle_vals,
    }
    # Dropout is a grid dimension too — expose it as a filter. Derive values from
    # the built records (not clean['dropout']) so NaN->0.0 fallbacks stay in sync
    # with each point's d.dropout, otherwise those points would filter out.
    if has_dropout:
        _fv['dropout'] = sorted({r['dropout'] for r in records})
    filter_values_json = _json.dumps(_fv)

    # HTML/JS fragments injected only when a dropout column exists.
    dropout_filter_group = ('''
        <div class="filter-group">
            <label>Dropout:</label>
            <div class="ms-wrap" id="msDropout"></div>
        </div>''' if has_dropout else '')
    dropout_init_js = ("initMultiSelect('msDropout', 'dropout');" if has_dropout else '')

    metric_options_html = ''
    for mc in metric_cols:
        metric_options_html += f'<option value="{mc}">{mc}</option>\n'

    html = f"""<!DOCTYPE html>
<html><head>
<title>Explorer v17: {metric_column}</title>
<!--PLOTLY_JS_PLACEHOLDER-->
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           margin: 0; padding: 20px; }}
    .controls {{ background: #f8f9fa; padding: 12px 20px; border-radius: 8px; margin-bottom: 8px; }}
    .controls h2 {{ margin: 0 0 8px 0; font-size: 18px; }}
    .filter-bar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 14px; }}
    .row2 {{ display: flex; flex-wrap: wrap; align-items: center; gap: 14px; margin-top: 10px;
             padding-top: 10px; border-top: 1px solid #ddd; }}
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
    .toggle-btn:disabled {{ background: #f0f0f0; color: #bbb; border-color: #e0e0e0;
                            cursor: not-allowed; }}
    .toggle-btn:disabled:hover {{ background: #f0f0f0; }}
    .btn {{ font-size: 12px; padding: 5px 12px; border: 1px solid #ccc; border-radius: 4px;
            background: white; cursor: pointer; }}
    .btn:hover {{ background: #e9ecef; }}
    .btn.active {{ background: #52a8ad; color: white; border-color: #52a8ad; }}
    /* Always visible. These used to be display:none until a scatter selection
       existed, which made them unfindable - the Export Selection button was in
       the DOM but invisible, so it read as missing. A control you cannot see is
       a control that does not exist. They stay muted until there IS a selection,
       and .has-selection lights them up. */
    .btn-ts {{ font-size: 12px; padding: 5px 12px; border: 1px solid #d9a066; border-radius: 4px;
               background: #fbf6f0; cursor: pointer; font-weight: 600; color: #b07a3c;
               display: inline-block; opacity: 0.75; }}
    .btn-ts:hover {{ background: #ff7f00; color: white; opacity: 1; }}
    .btn-ts.has-selection {{ display: inline-block; opacity: 1;
               border-color: #ff7f00; color: #ff7f00; background: #fff8f0; }}
    .ts-badge {{ font-size: 11px; padding: 2px 8px; border-radius: 10px;
                 background: #ff7f00; color: white; font-weight: 600; margin-left: 4px; }}
    .data-series {{ display: flex; align-items: center; gap: 4px; }}
    .data-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: middle; }}
    .remove-btn {{ font-size: 11px; padding: 2px 6px; border: 1px solid #ccc; border-radius: 3px;
                   background: white; cursor: pointer; color: #c0392b; font-weight: bold; }}
    .remove-btn:hover {{ background: #fce4e4; }}
    .add-btn {{ font-size: 13px; padding: 3px 10px; border: 1px solid #aaa; border-radius: 4px;
                background: white; cursor: pointer; font-weight: bold; color: #333; }}
    .add-btn:hover {{ background: #e9ecef; }}
    .ms-wrap {{ position: relative; display: inline-block; }}
    .ms-btn {{ font-size: 12px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px;
               background: white; cursor: pointer; min-width: 60px; text-align: left; }}
    .ms-btn:hover {{ background: #f5f5f5; }}
    .ms-panel {{ display: none; position: absolute; top: 100%; left: 0; z-index: 100;
                 background: white; border: 1px solid #ccc; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                 padding: 6px; min-width: 100px; max-height: 200px; overflow-y: auto; }}
    .ms-panel.open {{ display: block; }}
    .ms-panel label {{ display: block; font-size: 12px; padding: 2px 4px; cursor: pointer; white-space: nowrap; }}
    .ms-panel label:hover {{ background: #f0f7f7; }}
    .ms-panel input {{ margin-right: 4px; }}
    .ms-bulk {{ display: flex; gap: 6px; border-bottom: 1px solid #eee;
                padding-bottom: 5px; margin-bottom: 5px; }}
    .ms-bulk-btn {{ flex: 1; font-size: 11px; padding: 2px 6px; border: 1px solid #ccc;
                    border-radius: 3px; background: white; cursor: pointer; color: #333; }}
    .ms-bulk-btn:hover {{ background: #f0f7f7; border-color: #52a8ad; }}
    .row-adv {{ display: flex; flex-wrap: wrap; align-items: center; gap: 14px; margin-top: 10px;
                padding: 10px 12px; border: 1px dashed #cbd5d6; border-radius: 6px;
                background: #fbfdfd; }}
    /* Help overlay */
    .help-overlay {{ display: none; position: fixed; inset: 0; z-index: 1000;
                     background: rgba(0,0,0,0.45); }}
    .help-overlay.open {{ display: flex; align-items: center; justify-content: center; }}
    .help-modal {{ background: white; width: min(820px, 92vw); max-height: 86vh; display: flex;
                   flex-direction: column; border-radius: 10px; box-shadow: 0 10px 40px rgba(0,0,0,0.3); }}
    .help-head {{ display: flex; align-items: center; gap: 12px; padding: 14px 18px;
                  border-bottom: 1px solid #eee; }}
    .help-head h3 {{ margin: 0; font-size: 18px; flex-shrink: 0; }}
    .help-head input {{ flex: 1; font-size: 14px; padding: 6px 10px; border: 1px solid #ccc; border-radius: 6px; }}
    .help-body {{ padding: 8px 18px 18px; overflow-y: auto; }}
    .help-item {{ padding: 10px 0; border-bottom: 1px solid #f2f2f2; }}
    .help-item h4 {{ margin: 0 0 4px 0; font-size: 14px; color: #2a7a7f; }}
    .help-item p {{ margin: 0; font-size: 13px; color: #444; line-height: 1.5; }}
    .help-empty {{ color: #999; font-size: 13px; padding: 16px 0; }}
    /* Guided tour */
    .tour-backdrop {{ display: none; position: fixed; inset: 0; z-index: 1100;
                      background: rgba(0,0,0,0.55); }}
    .tour-highlight {{ display: none; position: absolute; z-index: 1101; border-radius: 6px;
                       box-shadow: 0 0 0 3px #52a8ad, 0 0 0 9999px rgba(0,0,0,0.55);
                       pointer-events: none; transition: all 0.15s ease; }}
    .tour-tip {{ display: none; position: absolute; z-index: 1102; background: white; border-radius: 8px;
                 box-shadow: 0 6px 24px rgba(0,0,0,0.35); padding: 14px 16px; width: 300px; }}
    .tour-tip-title {{ font-size: 15px; font-weight: 700; color: #2a7a7f; margin-bottom: 6px; }}
    .tour-tip-body {{ font-size: 13px; color: #444; line-height: 1.5; margin-bottom: 10px; }}
    .tour-tip-nav {{ display: flex; align-items: center; gap: 6px; }}
    .tour-progress {{ font-size: 12px; color: #999; }}
    #plotDiv {{ width: 100%; }}
    /* --- Dark mode: overrides applied when <body> has class 'dark' --- */
    body.dark {{ background: #1e1f22; color: #e3e3e3; }}
    body.dark .controls {{ background: #2b2d31; }}
    body.dark .controls h2 {{ color: #e3e3e3; }}
    body.dark .btn, body.dark .toggle-btn, body.dark .ms-btn, body.dark .ms-bulk-btn,
    body.dark .add-btn, body.dark .remove-btn {{ background: #3a3d42; color: #e3e3e3; border-color: #555; }}
    body.dark .btn:hover, body.dark .toggle-btn:hover, body.dark .ms-btn:hover,
    body.dark .ms-bulk-btn:hover, body.dark .add-btn:hover {{ background: #4a4e54; }}
    body.dark .btn.active, body.dark .toggle-btn.active {{ background: #52a8ad; color: #fff; border-color: #52a8ad; }}
    body.dark .ms-panel {{ background: #2b2d31; border-color: #555; box-shadow: 0 2px 8px rgba(0,0,0,0.6); }}
    body.dark .ms-panel label:hover, body.dark .ms-bulk-btn:hover {{ background: #363a40; }}
    body.dark select, body.dark input {{ background: #3a3d42; color: #e3e3e3; border-color: #555; }}
    body.dark .btn-ts {{ background: #3a2f22; }}
    body.dark .help-modal, body.dark .tour-tip {{ background: #2b2d31; color: #e3e3e3; }}
    body.dark .tour-tip-body {{ color: #cfcfcf; }}
    body.dark label {{ color: #e3e3e3; }}
</style>
</head><body>
<div class="controls">
    <h2>Hyperparameter Explorer <span style="font-size:12px; color:#999; font-weight:400;">v17</span></h2>
    <!-- Row 1: plot type, axes, data series, color-by, toggles -->
    <div class="filter-bar">
        <div class="filter-group">
            <button class="toggle-btn active" id="btnScatter" onclick="setPlotType('scatter')">Scatter</button>
            <button class="toggle-btn" id="btnBox" onclick="setPlotType('box')">Box</button>
            <button class="toggle-btn" id="btnViolin" onclick="setPlotType('violin')">Violin</button>
        </div>
        <div class="separator"></div>
        <div class="filter-group">
            <label>Show:</label>
            <button class="toggle-btn active" id="btnRaw" onclick="setShowMode('raw')">Raw</button>
            <button class="toggle-btn" id="btnAvg" onclick="setShowMode('avg')">Avg</button>
            <button class="toggle-btn" id="btnAvgSpread" onclick="setShowMode('avgspread')">Avg+Spread</button>
        </div>
        <div class="filter-group" id="spreadGroup" style="display:none;">
            <label>Spread:</label>
            <span style="display:flex; gap:8px; align-items:center;">
                <label style="font-weight:400;"><input type="checkbox" class="spreadCb" value="std" onchange="onSpreadModeChange()"> Std</label>
                <label style="font-weight:400;"><input type="checkbox" class="spreadCb" value="p5_95" checked onchange="onSpreadModeChange()"> 5-95</label>
                <label style="font-weight:400;"><input type="checkbox" class="spreadCb" value="p10_90" onchange="onSpreadModeChange()"> 10-90</label>
                <label style="font-weight:400;"><input type="checkbox" class="spreadCb" value="p0_100" onchange="onSpreadModeChange()"> 0-100</label>
            </span>
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
                <option value="combo">Combo (one per config)</option>
            </select>
        </div>
        <div class="filter-group">
            <label>Y Label:</label>
            <input id="yLabel" type="text" value="{metric_column}" onchange="rebuild()" />
        </div>
        <div class="separator"></div>
        <div id="dataSeriesContainer" style="display:flex; flex-wrap:wrap; align-items:center; gap:8px;"></div>
        <button class="add-btn" onclick="addDataSeries()" title="Add another metric">+</button>
        <div class="separator"></div>
        <div class="separator"></div>
        <button id="optionsBtn" class="btn" onclick="toggleOptions()" title="Show/hide secondary controls">&#9881; Options</button>
        <button id="helpBtn" class="btn" onclick="openHelp()" title="Searchable help">? Help</button>
        <button id="tourBtn" class="btn" onclick="startTour()" title="Guided walkthrough of every control">Take a tour</button>
    </div>
    <!-- Collapsible advanced options (hidden by default) -->
    <div class="row-adv" id="advPanel" style="display:none;">
        <div class="filter-group">
            <label style="font-size:12px;">Export file
                <input type="text" id="exportName" value="cove_selection.json"
                       style="font-size:12px; width:200px; padding:3px;"
                       title="Filename for Export Selection. Change it to start a separate list.">
            </label>
            <span id="basketCount" style="font-size:11px; color:#888;"></span>
            <button class="btn" onclick="clearExportList()"
                    title="Empty the accumulated selection list">Clear list</button>
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
                <option value="combo">Combo</option>
                <option value="cyc_recurrence">Cycle recurrence</option>
                <option value="lt_recurrence">Leadtime recurrence</option>
            </select>
        </div>
        <div class="filter-group">
            <label>Min cycles:</label>
            <select id="minCycles" onchange="rebuild()"><option value="1" selected>1</option></select>
        </div>
        <button id="boxmeanBtn" class="btn" style="display:none;" onclick="toggleBoxmean()">Mean+SD: OFF</button>
        <div class="filter-group" id="quartileGroup" style="display:none;">
            <label>Quartiles:</label>
            <select id="quartileSel" onchange="onQuartileChange()">
                <option value="linear" selected>Linear (25/75)</option>
                <option value="exclusive">Exclusive</option>
                <option value="inclusive">Inclusive</option>
                <option value="custom">Custom pct&hellip;</option>
            </select>
            <span id="customPctInputs" style="display:none; align-items:center; gap:4px;">
                box
                <input id="qLo" type="number" value="10" min="0" max="100" step="any" style="width:52px;" onchange="onQuartileChange()" title="Box lower percentile" />-
                <input id="qHi" type="number" value="90" min="0" max="100" step="any" style="width:52px;" onchange="onQuartileChange()" title="Box upper percentile" />
                whisk
                <input id="wLo" type="number" value="0" min="0" max="100" step="any" style="width:52px;" onchange="onQuartileChange()" title="Whisker lower percentile" />-
                <input id="wHi" type="number" value="100" min="0" max="100" step="any" style="width:52px;" onchange="onQuartileChange()" title="Whisker upper percentile" />
            </span>
        </div>
        <button id="offsetBtn" class="btn" onclick="toggleOffset()">Offset: OFF</button>
        <button id="pinYBtn" class="btn" onclick="togglePinY()" title="Lock the Y range so it stays fixed across redraws (e.g. when switching spread mode)">Pin Y: OFF</button>
        <span id="pinYInputs" class="filter-group" style="display:none;">
            <label>Y:</label>
            <input id="yMin" type="number" step="any" style="width:75px;" onchange="applyPinY()" title="Y min" />
            <input id="yMax" type="number" step="any" style="width:75px;" onchange="applyPinY()" title="Y max" />
        </span>
    </div>
    <!-- Row 2: legend, parameter filters, model count, send to TS -->
    <div class="row2">
        <button id="legendBtn" class="btn" onclick="toggleLegend()">Legend: OFF</button>
        <button id="darkBtn" class="btn" onclick="toggleDark()">Dark: OFF</button>
        <button id="repsBtn" class="btn" onclick="toggleReps()" title="In Avg+Spread: overlay the individual rep dots on top of the mean+spread">Reps: OFF</button>
        <button id="showAllBtn" class="btn" onclick="showAllTraces()" title="Un-hide every combo hidden by clicking or the legend">Show All</button>
        <div class="separator"></div>
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
            <label>Leadtime:</label>
            <div class="ms-wrap" id="msLT"></div>
        </div>
        <div class="filter-group">
            <label>Cycle:</label>
            <div class="ms-wrap" id="msCy"></div>
        </div>{dropout_filter_group}
        <div class="separator"></div>
        <span id="modelCount" class="model-count">Models (0)</span>
        <span id="missingCount" class="missing-count" style="display:none;">Missing (0)</span>
        <span id="comboWarn" class="missing-count" style="display:none;"></span>
        <div class="separator"></div>
        <button id="tsBtn" class="btn-ts" onclick="sendToTimeSeries()">Send to Time Series</button>
        <span id="tsBadge" style="display:none;" class="ts-badge">0</span>
        <button id="exportBtn" class="btn-ts" onclick="exportSelection()" title="Download the selected configs as a JSON file to email back">Export Selection</button>
    </div>
</div>
<div id="plotDiv"></div>

<!-- Help overlay (searchable reference) -->
<div id="helpOverlay" class="help-overlay">
  <div class="help-modal">
    <div class="help-head">
      <h3>Explorer Help</h3>
      <input id="helpSearch" type="text" placeholder="Search help&hellip;" oninput="renderHelp()" />
      <button class="btn" onclick="startTour()">Take a tour</button>
      <button class="btn" onclick="closeHelp()">Close</button>
    </div>
    <div id="helpBody" class="help-body"></div>
  </div>
</div>

<!-- Guided tour -->
<div id="tourBackdrop" class="tour-backdrop" onclick="endTour()"></div>
<div id="tourHighlight" class="tour-highlight"></div>
<div id="tourTip" class="tour-tip">
  <div id="tourTipTitle" class="tour-tip-title"></div>
  <div id="tourTipBody" class="tour-tip-body"></div>
  <div class="tour-tip-nav">
    <span id="tourProgress" class="tour-progress"></span>
    <span style="flex:1;"></span>
    <button class="btn" onclick="endTour()">Skip</button>
    <button class="btn" id="tourBackBtn" onclick="tourPrev()">Back</button>
    <button class="btn active" id="tourNextBtn" onclick="tourNext()">Next</button>
  </div>
</div>

<script>
var allData = {data_json};
var metricCols = {metric_cols_json};
var filterValues = {filter_values_json};
var metricOptionsHTML = '{metric_options_html.replace(chr(10), "").replace(chr(13), "")}';
var plotType = 'scatter';
var showBoxmean = false;
var showOffset = false;
var showLegend = false;
var showReps = false;

// v16: aggregation state
var showMode = 'raw';         // 'raw' | 'avg' | 'avgspread'
// Avg+Spread can show MULTIPLE fences at once (Option A: layered nested fences).
var spreadModes = ['p5_95'];  // any of: 'std','p5_95','p10_90','p0_100'

// Each spread level: how to derive [lo,hi] from a metric's rep-stats, plus its
// draw color. Ordered widest -> narrowest so nested fences read cleanly.
var SPREAD_DEFS = {{
    p0_100: {{ label: '0-100',  color: '#c9c9c9', bounds: function(s) {{ return [s.min, s.max]; }} }},
    p5_95:  {{ label: '5-95',   color: '#7fb3d5', bounds: function(s) {{ return [s.p5, s.p95]; }} }},
    p10_90: {{ label: '10-90',  color: '#2e86c1', bounds: function(s) {{ return [s.p10, s.p90]; }} }},
    std:    {{ label: 'Std',    color: '#1b2631', bounds: function(s) {{ return [s.mean - s.sd, s.mean + s.sd]; }} }}
}};
var SPREAD_ORDER = ['p0_100', 'p5_95', 'p10_90', 'std'];  // widest first (draw order)

// v16 item 3: quartile method for Box/Violin.
// 'linear'|'exclusive'|'inclusive' are Plotly built-ins; 'custom' uses precomputed
// percentiles (custom box only, single-metric box path).
var quartileMethod = 'linear';
var customPct = {{ qLo: 10, qHi: 90, wLo: 0, wHi: 100 }};  // box edges + whiskers (percentiles)

// v16: manual Y-axis pin (off by default; only locks when the user clicks)
var pinY = false;
var pinYRange = null;         // [min, max] when pinned

var dataSeriesCounter = 0;
var dataSeries = [];

// Track selected points from plotly_selected
var selectedPoints = [];

var seriesColors = [
    '#52a8ad', '#e05d5d', '#4daf4a', '#984ea3', '#ff7f00',
    '#377eb8', '#a65628', '#f781bf', '#66c2a5', '#fc8d62'
];

var xAxisLabels = {{
    'neurons': 'Hidden Units', 'num_layers': 'Number of Layers',
    'leadtime': 'Leadtime (h)', 'cycle': 'Cycle', 'activation': 'Activation Function',
    'combo': 'Combo', 'cyc_recurrence': 'Cycle recurrence', 'lt_recurrence': 'Leadtime recurrence'
}};

var colorPalette = [
    '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
    '#a65628','#f781bf','#66c2a5','#fc8d62','#8da0cb',
    '#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'
];

// --- v16: combo grouping + on-the-fly aggregation -----------------------------
function comboKey(d) {{
    // dropout IS part of a config's identity — omitting it merges dropout-variants
    // and undercounts combos in Avg mode.
    return d.activation + '|' + d.num_layers + '|' + d.neurons + '|' +
           d.leadtime + '|' + d.cycle + '|' + d.dropout;
}}

// Architecture identity = hyperparameters WITHOUT leadtime/cycle. Used to measure
// how broadly an arch recurs: a strong generalizer shows up across many cycles and
// leadtimes; a specialist appears in only one. dropout is part of the arch.
function archKey(d) {{
    return d.activation + '|' + d.num_layers + '|' + d.neurons + '|' + d.dropout;
}}

// archRecurrence[archKey] = {{ c: #distinct cycles, l: #distinct leadtimes }} over
// whatever row set it was last computed on (the current-filter set, per rebuild).
var archRecurrence = {{}};
function computeArchRecurrence(rows) {{
    var tmp = {{}};
    rows.forEach(function(d) {{
        var k = archKey(d);
        var e = tmp[k] || (tmp[k] = {{ cy: {{}}, lt: {{}} }});
        e.cy[d.cycle] = 1;
        e.lt[d.leadtime] = 1;
    }});
    archRecurrence = {{}};
    Object.keys(tmp).forEach(function(k) {{
        archRecurrence[k] = {{ c: Object.keys(tmp[k].cy).length,
                               l: Object.keys(tmp[k].lt).length }};
    }});
}}
function archRec(d) {{ return archRecurrence[archKey(d)] || {{ c: 0, l: 0 }}; }}

// Guardrail #1 data: how many reps does the fullest combo have?
var maxRepsPerCombo = (function() {{
    var counts = {{}};
    var mx = 0;
    allData.forEach(function(d) {{
        var k = comboKey(d);
        counts[k] = (counts[k] || 0) + 1;
        if (counts[k] > mx) mx = counts[k];
    }});
    return mx;
}})();
var aggEnabled = maxRepsPerCombo >= 2;

// linear-interpolated percentile over a sorted numeric array, p in [0,100]
function pctl(sorted, p) {{
    if (sorted.length === 0) return null;
    if (sorted.length === 1) return sorted[0];
    var idx = (p / 100) * (sorted.length - 1);
    var lo = Math.floor(idx), hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}}

// Collapse per-rep rows into one record per combo. Each record carries, per
// metric: the mean (as d[metric]) and a full stats object d['__stats_'+m] =
// {{mean, sd, min, max, p5, p95, p10, p90}} so ANY spread level can be derived
// on the fly. d.__n = rep count.
function aggregateRecords(rows) {{
    var groups = {{}};
    rows.forEach(function(d) {{
        var k = comboKey(d);
        (groups[k] = groups[k] || []).push(d);
    }});
    var out = [];
    Object.keys(groups).forEach(function(k) {{
        var g = groups[k];
        var first = g[0];
        var rec = {{
            activation: first.activation, num_layers: first.num_layers,
            neurons: first.neurons, leadtime: first.leadtime, cycle: first.cycle,
            dropout: first.dropout, combo: first.combo,
            __n: g.length
        }};
        metricCols.forEach(function(mc) {{
            var vals = [];
            g.forEach(function(d) {{ if (d[mc] != null) vals.push(d[mc]); }});
            if (vals.length === 0) {{ rec[mc] = null; rec['__stats_' + mc] = null; return; }}
            var sum = 0; vals.forEach(function(v) {{ sum += v; }});
            var mean = sum / vals.length;
            var ss = 0; vals.forEach(function(v) {{ ss += (v - mean) * (v - mean); }});
            var sorted = vals.slice().sort(function(a, b) {{ return a - b; }});
            rec[mc] = mean;
            rec['__stats_' + mc] = {{
                mean: mean, sd: Math.sqrt(ss / vals.length),
                min: sorted[0], max: sorted[sorted.length - 1],
                p5: pctl(sorted, 5), p95: pctl(sorted, 95),
                p10: pctl(sorted, 10), p90: pctl(sorted, 90)
            }};
        }});
        out.push(rec);
    }});
    return out;
}}

// [lo, hi] for one spread level from a metric's stats object.
function boundsFor(stats, modeKey) {{
    if (!stats) return null;
    return SPREAD_DEFS[modeKey].bounds(stats);
}}

// error_y spec for a data list at one spread level (Avg+Spread scatter, Option A).
function errBarsForMode(list, metric, modeKey) {{
    return {{
        type: 'data', symmetric: false,
        array: list.map(function(d) {{
            var b = boundsFor(d['__stats_' + metric], modeKey);
            return (b && d[metric] != null) ? b[1] - d[metric] : 0;
        }}),
        arrayminus: list.map(function(d) {{
            var b = boundsFor(d['__stats_' + metric], modeKey);
            return (b && d[metric] != null) ? d[metric] - b[0] : 0;
        }}),
        thickness: 1.4, width: 3, color: SPREAD_DEFS[modeKey].color
    }};
}}

// Option A: mean dots + one nested fence layer per selected spread level.
// colorBy === 'none': pooled fences + one mean trace (original v16 behavior).
// colorBy active (e.g. combo): one legend entry PER group, with each group's mean
// dot + all its nested fences sharing a legendgroup — so clicking the legend entry
// toggles the whole group at once. This enables the same "cull bad combos via the
// legend" workflow inside the mean+std view. (Offset is still ignored: categorical X.)
function buildAvgSpreadScatter(list, metric, xParam, colorBy, rawReps, showReps) {{
    var valid = list.filter(function(d) {{ return d[metric] != null; }});
    var traces = [];

    if (!colorBy || colorBy === 'none') {{
        var xs = valid.map(function(d) {{ return String(d[xParam]); }});
        var ys = valid.map(function(d) {{ return d[metric]; }});
        // fence layers first (widest -> narrowest) so narrower ones sit on top
        SPREAD_ORDER.forEach(function(modeKey) {{
            if (spreadModes.indexOf(modeKey) === -1) return;
            traces.push({{
                x: xs, y: ys, error_y: errBarsForMode(valid, metric, modeKey),
                mode: 'markers', type: 'scatter', name: SPREAD_DEFS[modeKey].label,
                marker: {{ size: 0.1, color: SPREAD_DEFS[modeKey].color }},
                hoverinfo: 'skip', showlegend: true
            }});
        }});
        // individual rep dots (behind the mean dot) so you see spread + models at once
        if (showReps && rawReps) {{
            var reps = rawReps.filter(function(d) {{ return d[metric] != null; }});
            traces.push({{
                x: reps.map(function(d) {{ return String(d[xParam]); }}),
                y: reps.map(function(d) {{ return d[metric]; }}),
                mode: 'markers', type: 'scatter', name: 'reps',
                marker: {{ size: 5, color: 'rgba(120,120,120,0.45)',
                           line: {{ width: 0.5, color: 'rgba(0,0,0,0.2)' }} }},
                text: reps.map(function(d) {{ return makeHover(d, metric); }}),
                hoverinfo: 'text', showlegend: true, customdata: reps
            }});
        }}
        // mean dots on top
        traces.push({{
            x: xs, y: ys, mode: 'markers', type: 'scatter', name: 'mean',
            marker: {{ size: 7, color: '#333', line: {{ width: 0.5, color: 'rgba(0,0,0,0.25)' }} }},
            text: valid.map(function(d) {{ return makeHover(d, metric); }}),
            hoverinfo: 'text', showlegend: true, customdata: valid
        }});
        return traces;
    }}

    // --- color-by active: per-group mean + fences, grouped for one-click cull ---
    var groups = {{}};
    valid.forEach(function(d) {{
        var k = String(d[colorBy]);
        (groups[k] = groups[k] || []).push(d);
    }});
    var keys = Object.keys(groups);
    if (colorBy !== 'activation') {{
        keys.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
    }} else {{ keys.sort(); }}

    // Group the raw reps by the same key so the overlay can be per-combo (and thus
    // share each combo's legendgroup, so culling a combo hides its rep dots too).
    var repGroups = {{}};
    if (showReps && rawReps) {{
        rawReps.forEach(function(d) {{
            if (d[metric] == null) return;
            var k = String(d[colorBy]);
            (repGroups[k] = repGroups[k] || []).push(d);
        }});
    }}

    keys.forEach(function(key, ci) {{
        var g = groups[key];
        var col = colorPalette[ci % colorPalette.length];
        var gx = g.map(function(d) {{ return String(d[xParam]); }});
        var gy = g.map(function(d) {{ return d[metric]; }});
        // fences for this group: no own legend entry, but same legendgroup so the
        // group's single legend toggle hides them too. errBarsForMode returns a
        // fresh object each call, so recoloring it to the combo's color is safe.
        SPREAD_ORDER.forEach(function(modeKey) {{
            if (spreadModes.indexOf(modeKey) === -1) return;
            var eb = errBarsForMode(g, metric, modeKey);
            eb.color = col;
            traces.push({{
                x: gx, y: gy, error_y: eb,
                mode: 'markers', type: 'scatter', name: key,
                legendgroup: key, showlegend: false,
                marker: {{ size: 0.1, color: col }},
                hoverinfo: 'skip'
            }});
        }});
        // rep dots for this combo (behind mean, same legendgroup so cull hides them)
        var rg = repGroups[key];
        if (rg && rg.length) {{
            traces.push({{
                x: rg.map(function(d) {{ return String(d[xParam]); }}),
                y: rg.map(function(d) {{ return d[metric]; }}),
                mode: 'markers', type: 'scatter', name: key,
                legendgroup: key, showlegend: false,
                marker: {{ size: 5, color: col, opacity: 0.4,
                           line: {{ width: 0.5, color: 'rgba(0,0,0,0.2)' }} }},
                text: rg.map(function(d) {{ return makeHover(d, metric); }}),
                hoverinfo: 'text', customdata: rg
            }});
        }}
        // mean dot on top carries the group's one legend entry
        traces.push({{
            x: gx, y: gy, mode: 'markers', type: 'scatter', name: key,
            legendgroup: key, showlegend: true,
            marker: {{ size: 7, color: col, line: {{ width: 0.5, color: 'rgba(0,0,0,0.25)' }} }},
            text: g.map(function(d) {{ return makeHover(d, metric); }}),
            hoverinfo: 'text', customdata: g
        }});
    }});
    return traces;
}}

// 'custom' is not a Plotly quartilemethod — anywhere we feed raw y (multi-metric
// or color-by boxes, where custom precomputed boxes aren't supported), fall back
// to 'linear'. Custom is honored only in the single-metric box path.
function effectiveQuartileMethod() {{
    return (quartileMethod === 'custom') ? 'linear' : quartileMethod;
}}

// Precomputed box stats for one array of values, honoring the custom-percentile
// choice (item 3). Used only when quartileMethod === 'custom'.
function customBoxStats(vals) {{
    var s = vals.slice().sort(function(a, b) {{ return a - b; }});
    var sum = 0; vals.forEach(function(v) {{ sum += v; }});
    return {{
        q1: pctl(s, customPct.qLo), median: pctl(s, 50), q3: pctl(s, customPct.qHi),
        lowerfence: pctl(s, customPct.wLo), upperfence: pctl(s, customPct.wHi),
        mean: sum / vals.length
    }};
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

    // Select all / deselect all
    var bulk = document.createElement('div');
    bulk.className = 'ms-bulk';
    ['All', 'None'].forEach(function(act) {{
        var bb = document.createElement('button');
        bb.type = 'button';
        bb.className = 'ms-bulk-btn';
        bb.textContent = act;
        bb.addEventListener('click', function(e) {{
            e.stopPropagation();
            var check = (act === 'All');
            panel.querySelectorAll('input[type="checkbox"]').forEach(function(cb) {{ cb.checked = check; }});
            updateMsState(filterKey, wrapId);
        }});
        bulk.appendChild(bb);
    }});
    panel.appendChild(bulk);

    values.forEach(function(v) {{
        var lbl = document.createElement('label');
        var cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = true;
        cb.value = String(v);
        cb.dataset.filterKey = filterKey;
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

    rebuild();
}}

// Close open filter panels when clicking OUTSIDE any filter widget. Clicks inside
// a widget (checkboxes AND the All/None buttons) are left to their own handlers.
// (Previously a capture-phase stopPropagation swallowed clicks inside .ms-panel,
// which killed the All/None buttons — checkboxes survived only because they use
// 'change', not 'click'.)
document.addEventListener('click', function(e) {{
    if (e.target.closest('.ms-wrap')) return;
    document.querySelectorAll('.ms-panel.open').forEach(function(p) {{
        p.classList.remove('open');
    }});
}});

// --- Data series ---
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
    if (dataSeries.length === 1) {{
        document.getElementById('yLabel').value = metric;
    }}
    rebuild();
}}

function updateColorByVisibility() {{
    var multi = dataSeries.length > 1;
    document.getElementById('colorByGroup').style.display = multi ? 'none' : '';
    if (multi) document.getElementById('colorBy').value = 'none';
}}

// --- Controls ---
function setPlotType(type) {{
    plotType = type;
    document.getElementById('btnScatter').classList.toggle('active', type === 'scatter');
    document.getElementById('btnBox').classList.toggle('active', type === 'box');
    document.getElementById('btnViolin').classList.toggle('active', type === 'violin');
    document.getElementById('boxmeanBtn').style.display = (type === 'box') ? '' : 'none';
    // quartile control only applies to Box/Violin
    document.getElementById('quartileGroup').style.display =
        (type === 'box' || type === 'violin') ? '' : 'none';
    rebuild();
}}

function onQuartileChange() {{
    quartileMethod = document.getElementById('quartileSel').value;
    document.getElementById('customPctInputs').style.display =
        (quartileMethod === 'custom') ? '' : 'none';
    customPct.qLo = parseFloat(document.getElementById('qLo').value);
    customPct.qHi = parseFloat(document.getElementById('qHi').value);
    customPct.wLo = parseFloat(document.getElementById('wLo').value);
    customPct.wHi = parseFloat(document.getElementById('wHi').value);
    rebuild();
}}

function setShowMode(mode) {{
    // Guardrail #1: aggregation needs per-rep data.
    if ((mode === 'avg' || mode === 'avgspread') && !aggEnabled) return;
    showMode = mode;
    document.getElementById('btnRaw').classList.toggle('active', mode === 'raw');
    document.getElementById('btnAvg').classList.toggle('active', mode === 'avg');
    document.getElementById('btnAvgSpread').classList.toggle('active', mode === 'avgspread');
    document.getElementById('spreadGroup').style.display = (mode === 'avgspread') ? '' : 'none';
    rebuild();
}}

function onSpreadModeChange() {{
    var boxes = document.querySelectorAll('.spreadCb');
    spreadModes = [];
    boxes.forEach(function(cb) {{ if (cb.checked) spreadModes.push(cb.value); }});
    rebuild();
}}

function toggleBoxmean() {{
    showBoxmean = !showBoxmean;
    var btn = document.getElementById('boxmeanBtn');
    btn.textContent = showBoxmean ? 'Mean+SD: ON' : 'Mean+SD: OFF';
    btn.classList.toggle('active', showBoxmean);
    rebuild();
}}

function toggleOffset() {{
    showOffset = !showOffset;
    var btn = document.getElementById('offsetBtn');
    btn.textContent = showOffset ? 'Offset: ON' : 'Offset: OFF';
    btn.classList.toggle('active', showOffset);
    rebuild();
}}

function togglePinY() {{
    pinY = !pinY;
    var btn = document.getElementById('pinYBtn');
    var inputs = document.getElementById('pinYInputs');
    if (pinY) {{
        // Capture the range Plotly is currently showing, so the pin starts from
        // exactly what's on screen.
        var pd = document.getElementById('plotDiv');
        var r = (pd && pd._fullLayout && pd._fullLayout.yaxis) ? pd._fullLayout.yaxis.range : null;
        if (r) pinYRange = [r[0], r[1]];
        if (pinYRange) {{
            document.getElementById('yMin').value = pinYRange[0];
            document.getElementById('yMax').value = pinYRange[1];
        }}
        inputs.style.display = '';
        btn.textContent = 'Pin Y: ON';
        btn.classList.add('active');
    }} else {{
        inputs.style.display = 'none';
        pinYRange = null;
        btn.textContent = 'Pin Y: OFF';
        btn.classList.remove('active');
    }}
    rebuild();
}}

function applyPinY() {{
    var mn = parseFloat(document.getElementById('yMin').value);
    var mx = parseFloat(document.getElementById('yMax').value);
    if (!isNaN(mn) && !isNaN(mx)) {{
        pinYRange = [mn, mx];
        rebuild();
    }}
}}

// --- Collapsible options panel ---
function toggleOptions() {{
    var p = document.getElementById('advPanel');
    var open = (p.style.display === 'none');
    p.style.display = open ? '' : 'none';
    document.getElementById('optionsBtn').classList.toggle('active', open);
}}

// --- Help + guided tour (single source of truth) ---
// Each entry documents one control. `sel` (a CSS selector) makes it a tour stop;
// entries without `sel` still appear in the searchable help.
var helpSteps = [
    {{ sel: '#btnScatter', title: 'Plot type',
       body: 'Switch between Scatter, Box, and Violin. Scatter shows one dot per point; Box and Violin summarize the distribution at each X value.' }},
    {{ sel: '#btnRaw', title: 'Show mode: Raw / Avg / Avg+Spread',
       body: 'Raw plots every repetition. Avg collapses each config\\'s reps to one mean dot. Avg+Spread adds an error fence. Avg/Avg+Spread are disabled if the data has only one row per combo (an already-averaged file).' }},
    {{ sel: '#btnAvgSpread', title: 'Spread fences (layered)',
       body: 'In Avg+Spread mode you can tick MULTIPLE fences at once - Std, 5-95, 10-90, 0-100 - and they draw as nested error bars on each mean dot (widest to narrowest), so you can compare spread definitions side by side without swapping.' }},
    {{ sel: '#xAxis', title: 'Combo X axis (per-config boxes)',
       body: 'Set X axis to "Combo" and switch to Box/Violin to get one box per configuration (rather than pooled by hyperparameter). Filter to a single leadtime first so it stays readable; a warning appears above ~120 combos.' }},
    {{ sel: null, title: 'Quartile method (Box/Violin)',
       body: 'In the Options panel, choose how box quartiles are computed: Linear (default 25/75), Exclusive, or Inclusive (Plotly built-ins), or Custom to set your own box-edge and whisker percentiles (e.g. 10-90 box with 0-100 whiskers). Custom applies to single-metric boxes.' }},
    {{ sel: '#xAxis', title: 'X axis',
       body: 'Pick the hyperparameter on the X axis: hidden units, number of layers, leadtime, cycle, or activation.' }},
    {{ sel: '#yLabel', title: 'Y label',
       body: 'Rename the Y axis label. It auto-fills with the selected metric.' }},
    {{ sel: '#dataSeriesContainer', title: 'Data series',
       body: 'Choose which metric to plot. Use the + button to overlay several metrics at once (color-by is disabled while comparing multiple metrics).' }},
    {{ sel: '#optionsBtn', title: 'Options panel',
       body: 'Show or hide the secondary controls below: color-by, offset, Mean+SD (box), and Pin Y.' }},
    {{ sel: '#colorByGroup', title: 'Color by',
       body: 'Color the dots or boxes by a hyperparameter. Hidden when plotting multiple metrics.' }},
    {{ sel: '#offsetBtn', title: 'Offset',
       body: 'Spread overlapping points/boxes sideways so groups at the same X value do not sit on top of each other.' }},
    {{ sel: '#pinYBtn', title: 'Pin Y',
       body: 'Lock the Y range so it stays fixed across redraws - handy when switching spread modes to compare on the same scale. Off by default; after pinning you can edit the exact min/max.' }},
    {{ sel: '#btnBox', title: 'How the box is computed',
       body: 'Boxes use Plotly defaults: Q1/Q3 = 25th/75th percentile (linear method), median = 50th, whiskers = 1.5x IQR, points beyond drawn as outliers. Hover a box to read the exact values.' }},
    {{ sel: '#msAct', title: 'Filters',
       body: 'Multi-select checkboxes narrow the data by activation, layers, neurons, leadtime, and cycle. Use All / None to select or clear a filter quickly.' }},
    {{ sel: '#modelCount', title: 'Counts',
       body: 'Shows how many points (Raw) or combos (Avg) are currently plotted, plus any dropped for a missing metric value.' }},
    {{ sel: '#legendBtn', title: 'Legend',
       body: 'Show or hide the chart legend.' }},
    {{ sel: null, title: 'Send to Time Series',
       body: 'In Scatter, drag a box or lasso to select dots, then click Send to Time Series to open those exact models in the time-series comparison view.' }}
];

function openHelp() {{
    document.getElementById('helpOverlay').classList.add('open');
    document.getElementById('helpSearch').value = '';
    renderHelp();
    document.getElementById('helpSearch').focus();
}}
function closeHelp() {{
    document.getElementById('helpOverlay').classList.remove('open');
}}
function renderHelp() {{
    var q = (document.getElementById('helpSearch').value || '').toLowerCase();
    var body = document.getElementById('helpBody');
    var items = helpSteps.filter(function(s) {{
        return !q || (s.title + ' ' + s.body).toLowerCase().indexOf(q) !== -1;
    }});
    if (items.length === 0) {{
        body.innerHTML = '<div class="help-empty">No help topics match your search.</div>';
        return;
    }}
    body.innerHTML = items.map(function(s) {{
        return '<div class="help-item"><h4>' + s.title + '</h4><p>' + s.body + '</p></div>';
    }}).join('');
}}
// close help when clicking the dark backdrop (outside the modal)
document.getElementById('helpOverlay').addEventListener('click', function(e) {{
    if (e.target === this) closeHelp();
}});

var tourSteps = helpSteps.filter(function(s) {{ return s.sel; }});
var tourIdx = 0;
var _advWasOpen = false;

function startTour() {{
    closeHelp();
    var p = document.getElementById('advPanel');
    _advWasOpen = (p.style.display !== 'none');
    p.style.display = '';  // ensure advanced controls are visible during the tour
    document.getElementById('optionsBtn').classList.add('active');
    tourIdx = 0;
    document.getElementById('tourBackdrop').style.display = 'block';
    tourShow();
}}
function endTour() {{
    document.getElementById('tourBackdrop').style.display = 'none';
    document.getElementById('tourHighlight').style.display = 'none';
    document.getElementById('tourTip').style.display = 'none';
    if (!_advWasOpen) {{
        document.getElementById('advPanel').style.display = 'none';
        document.getElementById('optionsBtn').classList.remove('active');
    }}
}}
function tourNext() {{ if (tourIdx < tourSteps.length - 1) {{ tourIdx++; tourShow(); }} else {{ endTour(); }} }}
function tourPrev() {{ if (tourIdx > 0) {{ tourIdx--; tourShow(); }} }}
function tourShow() {{
    var step = tourSteps[tourIdx];
    var el = document.querySelector(step.sel);
    if (!el) {{ tourNext(); return; }}
    document.getElementById('tourTipTitle').textContent = step.title;
    document.getElementById('tourTipBody').innerHTML = step.body;
    document.getElementById('tourProgress').textContent = (tourIdx + 1) + ' / ' + tourSteps.length;
    document.getElementById('tourBackBtn').style.visibility = (tourIdx === 0) ? 'hidden' : 'visible';
    document.getElementById('tourNextBtn').textContent = (tourIdx === tourSteps.length - 1) ? 'Done' : 'Next';

    el.scrollIntoView({{ block: 'center', behavior: 'smooth' }});
    var r = el.getBoundingClientRect();
    var pad = 6;
    var hl = document.getElementById('tourHighlight');
    hl.style.display = 'block';
    hl.style.top = (r.top + window.scrollY - pad) + 'px';
    hl.style.left = (r.left + window.scrollX - pad) + 'px';
    hl.style.width = (r.width + pad * 2) + 'px';
    hl.style.height = (r.height + pad * 2) + 'px';

    var tip = document.getElementById('tourTip');
    tip.style.display = 'block';
    var vw = document.documentElement.clientWidth;
    var vh = document.documentElement.clientHeight;
    var tipLeft = r.left + window.scrollX;
    if (tipLeft > window.scrollX + vw - 320) tipLeft = window.scrollX + vw - 320;
    if (tipLeft < window.scrollX + 8) tipLeft = window.scrollX + 8;
    var tipTop;
    if (r.bottom + 190 > vh) tipTop = r.top + window.scrollY - 10 - tip.offsetHeight;
    else tipTop = r.bottom + window.scrollY + 10;
    tip.style.top = tipTop + 'px';
    tip.style.left = tipLeft + 'px';
}}
// Escape closes help or tour; reposition the tour on resize.
document.addEventListener('keydown', function(e) {{
    if (e.key !== 'Escape') return;
    if (document.getElementById('helpOverlay').classList.contains('open')) closeHelp();
    else if (document.getElementById('tourBackdrop').style.display === 'block') endTour();
}});

function toggleLegend() {{
    showLegend = !showLegend;
    var btn = document.getElementById('legendBtn');
    btn.textContent = showLegend ? 'Legend: ON' : 'Legend: OFF';
    btn.classList.toggle('active', showLegend);
    Plotly.relayout('plotDiv', {{'showlegend': showLegend}});
}}

function toggleReps() {{
    showReps = !showReps;
    var btn = document.getElementById('repsBtn');
    btn.textContent = showReps ? 'Reps: ON' : 'Reps: OFF';
    btn.classList.toggle('active', showReps);
    rebuild();
}}

// Un-hide every trace hidden via click-to-hide or the legend. The subsequent
// plotly_restyle fires Item B, which restores the full x-axis category set.
function showAllTraces() {{
    var gd = document.getElementById('plotDiv');
    var n = (gd.data || []).length;
    if (!n) return;
    var idxs = [];
    for (var i = 0; i < n; i++) idxs.push(i);
    Plotly.restyle(gd, {{ visible: true }}, idxs);
}}

var darkMode = false;
function toggleDark() {{
    darkMode = !darkMode;
    document.body.classList.toggle('dark', darkMode);
    var btn = document.getElementById('darkBtn');
    btn.textContent = darkMode ? 'Dark: ON' : 'Dark: OFF';
    btn.classList.toggle('active', darkMode);
    // rebuild so the plot picks up the dark paper/plot backgrounds + font colors
    rebuild();
}}

function applyFilter() {{
    return allData.filter(function(d) {{
        if (msState.activation.indexOf(d.activation) === -1) return false;
        if (msState.num_layers.indexOf(String(d.num_layers)) === -1) return false;
        if (msState.neurons.indexOf(String(d.neurons)) === -1) return false;
        if (msState.leadtime.indexOf(String(d.leadtime)) === -1) return false;
        if (msState.cycle.indexOf(String(d.cycle)) === -1) return false;
        if (msState.dropout && msState.dropout.indexOf(String(d.dropout)) === -1) return false;
        return true;
    }});
}}

function makeHover(d, metric) {{
    var s = 'Act: ' + d.activation + '<br>Layers: ' + d.num_layers +
            '<br>Neurons: ' + d.neurons + '<br>LT: ' + d.leadtime + 'h' +
            '<br>Cycle: ' + d.cycle +
            (d.dropout != null ? '<br>Dropout: ' + d.dropout : '') +
            // Raw mode only: an aggregated dot is a mean OVER reps, so a single
            // rep number there would name one of its inputs and read as though
            // the dot were that rep.
            (showMode === 'raw' && d.run_num != null ? '<br>Rep: ' + d.run_num : '') +
            '<br>' + metric + ': ' +
            (d[metric] != null ? d[metric].toFixed(4) : 'N/A');
    // Recurrence of this arch (act/layers/neurons/dropout) over the current filter.
    var r = archRec(d);
    s += '<br><i>arch in ' + r.c + ' cycle(s), ' + r.l + ' leadtime(s)</i>';
    // Guardrail #2: be explicit that this dot is a mean, and over how many reps.
    if (showMode !== 'raw' && d.__n != null) {{
        s += '<br><i>mean of ' + d.__n + ' reps</i>';
        if (showMode === 'avgspread') {{
            var stats = d['__stats_' + metric];
            SPREAD_ORDER.forEach(function(modeKey) {{
                if (spreadModes.indexOf(modeKey) === -1) return;
                var b = boundsFor(stats, modeKey);
                if (b) s += '<br>' + SPREAD_DEFS[modeKey].label + ': ' +
                            b[0].toFixed(4) + ' - ' + b[1].toFixed(4);
            }});
        }}
    }}
    return s;
}}

// --- Send to Time Series via localStorage ---
function updateTsButton() {{
    var btn = document.getElementById('tsBtn');
    var badge = document.getElementById('tsBadge');
    if (selectedPoints.length > 0) {{
        btn.classList.add('has-selection');
        badge.style.display = 'inline';
        badge.textContent = selectedPoints.length;
    }} else {{
        btn.classList.remove('has-selection');
        badge.style.display = 'none';
    }}
}}

function sendToTimeSeries() {{
    // No early return. This used to do NOTHING when nothing was lasso-selected,
    // which reads as a broken button - and the filter panels never populate
    // selectedPoints, so picking a combo over there left it empty too. Opening
    // the time-series page with an empty selection is a valid destination: it
    // lands blank with its own pickers ready.

    // Build filter criteria from selected points
    var models = selectedPoints.map(function(d) {{
        return {{
            activation: d.activation,
            num_layers: d.num_layers,
            neurons: d.neurons,
            leadtime: d.leadtime,
            cycle: d.cycle,
            // dropout is a grid axis: two configs identical apart from it are
            // DIFFERENT models. Leaving it out here merged them into one key and
            // sent an ambiguous selection downstream.
            dropout: d.dropout
        }};
    }});

    // Deduplicate by stringifying
    var seen = {{}};
    var unique = [];
    models.forEach(function(m) {{
        var key = m.activation + '_' + m.num_layers + '_' + m.neurons + '_' + m.leadtime + '_' + m.cycle + '_d' + m.dropout;
        if (!seen[key]) {{
            seen[key] = true;
            unique.push(m);
        }}
    }});

    localStorage.setItem('tsModels', JSON.stringify(unique));
    localStorage.setItem('tsTimestamp', Date.now().toString());

    // Open the time-series page in a new tab. cove_timeseries.html carries its
    // OWN prediction data, so it needs no server - it works from an emailed copy
    // opened over file://, which the old 05_timeseries.html + serve_timeseries
    // pair could not do.
    var base = window.location.href.replace(/\\/[^\\/]*$/, '/');
    window.open(base + 'cove_timeseries.html', '_blank');
}}

// --- Export Selection: download selected configs as a portable JSON ----------
// Pure client-side (Blob download) so it works from an emailed file opened via
// file:// with no server. The recipient emails this JSON back; generate_predictions
// --selection resolves each config to its .keras and bakes a self-contained TS html.
function exportSelection() {{
    if (selectedPoints.length === 0) {{
        alert('Select some dots first (drag a box or lasso in the Scatter view), then Export Selection.');
        return;
    }}
    // Dedup by combo identity PLUS rep: in raw mode each rep is its own point,
    // so keying on the combo alone would silently drop all but one of five
    // selected dots.
    var seen = {{}};
    var configs = [];
    selectedPoints.forEach(function(d) {{
        var k = comboKey(d) + '|' + (d.run_num != null ? d.run_num : '');
        if (seen[k]) return;
        seen[k] = true;
        configs.push({{
            model_type: d.model_type != null ? d.model_type : 'MAPE',
            activation: d.activation,
            num_layers: d.num_layers,
            neurons: d.neurons,
            leadtime: d.leadtime,
            cycle: d.cycle,
            dropout: d.dropout,
            run_num: d.run_num != null ? d.run_num : null
        }});
    }});
    var payload = {{
        schema_version: 1,
        exported_at: new Date().toISOString(),
        n_configs: configs.length,
        configs: configs
    }};
    writeExport(configs);
}}

// --- Accumulating export ------------------------------------------------------
// Two browser facts shape this, and neither is a choice we get to make:
//
//   1. A page cannot pick or remember a SAVE DIRECTORY with a normal download.
//      "Ask where to save each file" is a Chrome setting (Settings > Downloads).
//      What a page CAN do is keep a file HANDLE via showSaveFilePicker: you pick
//      the location once, and every later export writes to that same file with
//      no prompt. That is used here when the browser supports it.
//   2. A page cannot append to a file on disk. So the accumulated list is kept
//      in localStorage and the file is REWRITTEN complete each time. The visible
//      result is what you asked for - the file grows as you add selections -
//      without pretending we can append.
var _exportHandle = null;   // remembered for the session once you pick a spot

function loadExportList() {{
    try {{ return JSON.parse(localStorage.getItem('coveExportList') || '[]'); }}
    catch (e) {{ return []; }}
}}

function saveExportList(list) {{
    try {{ localStorage.setItem('coveExportList', JSON.stringify(list)); }}
    catch (e) {{ /* private mode: the download still works, just not the memory */ }}
    updateBasketCount();
}}

function updateBasketCount() {{
    var el = document.getElementById('basketCount');
    if (el) el.textContent = loadExportList().length + ' saved';
}}

function clearExportList() {{
    saveExportList([]);
    alert('Export list cleared. The next export starts a fresh file.');
}}

function writeExport(newConfigs) {{
    // Merge into the running list, de-duplicated on full combo identity so
    // exporting the same dots twice does not double them up.
    var list = loadExportList();
    var seen = {{}};
    list.forEach(function(c) {{ seen[exportKey(c)] = true; }});
    var added = 0;
    newConfigs.forEach(function(c) {{
        var k = exportKey(c);
        if (seen[k]) return;
        seen[k] = true;
        list.push(c);
        added++;
    }});
    saveExportList(list);

    var payload = {{
        schema_version: 1,
        exported_at: new Date().toISOString(),
        n_configs: list.length,
        configs: list
    }};
    var text = JSON.stringify(payload, null, 2);
    var name = (document.getElementById('exportName') || {{}}).value
               || 'cove_selection.json';

    if (window.showSaveFilePicker) {{
        saveViaHandle(text, name, added, list.length);
    }} else {{
        downloadFallback(text, name);
        alert('Added ' + added + ' new config(s). File now holds ' + list.length + '.');
    }}
}}

function exportKey(c) {{
    return [c.model_type, c.leadtime, c.cycle, c.activation,
            c.num_layers, c.neurons, c.dropout, c.run_num].join('|');
}}

async function saveViaHandle(text, name, added, total) {{
    try {{
        if (!_exportHandle) {{
            _exportHandle = await window.showSaveFilePicker({{
                suggestedName: name,
                types: [{{ description: 'JSON', accept: {{'application/json': ['.json']}} }}]
            }});
        }}
        var w = await _exportHandle.createWritable();
        await w.write(text);
        await w.close();
        alert('Added ' + added + ' new config(s). File now holds ' + total + '.');
    }} catch (e) {{
        if (e && e.name === 'AbortError') return;   // user cancelled the picker
        // Handle went stale (file moved/deleted) or the API refused: forget it
        // and fall back rather than silently writing nowhere.
        _exportHandle = null;
        downloadFallback(text, name);
    }}
}}

function downloadFallback(text, name) {{
    var blob = new Blob([text], {{type: 'application/json'}});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}}

// --- Trace builders ---
function buildMultiSeriesTraces(filtered, metrics, xParam) {{
    var nMetrics = metrics.length;
    var traces = [];

    var allXRaw = filtered.map(function(d) {{ return String(d[xParam]); }});
    var uniqueX = [...new Set(allXRaw)];
    if (xParam !== 'activation' && xParam !== 'combo') {{
        uniqueX.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
    }} else {{ uniqueX.sort(); }}

    if (plotType === 'scatter') {{
        var xMap = {{}};
        uniqueX.forEach(function(v, i) {{ xMap[v] = i; }});

        var spread = 0.6;
        var step = nMetrics > 1 ? spread / (nMetrics - 1) : 0;
        var startOffset = nMetrics > 1 ? -spread / 2 : 0;

        metrics.forEach(function(metric, mi) {{
            var color = getSeriesColor(mi);
            var offset = startOffset + mi * step;
            var validData = filtered.filter(function(d) {{ return d[metric] != null; }});

            if (nMetrics === 1 && !showOffset) {{
                traces.push({{
                    x: validData.map(function(d) {{ return String(d[xParam]); }}),
                    y: validData.map(function(d) {{ return d[metric]; }}),
                    mode: 'markers', type: 'scatter', name: metric,
                    marker: {{ size: 6, color: color, opacity: 0.55,
                               line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }} }},
                    text: validData.map(function(d) {{ return makeHover(d, metric); }}),
                    hoverinfo: 'text', showlegend: true,
                    customdata: validData
                }});
            }} else {{
                var xNums = validData.map(function(d) {{
                    return xMap[String(d[xParam])] + offset + (Math.random() - 0.5) * 0.08;
                }});
                traces.push({{
                    x: xNums,
                    y: validData.map(function(d) {{ return d[metric]; }}),
                    mode: 'markers', type: 'scatter', name: metric,
                    marker: {{ size: 6, color: color, opacity: 0.6,
                               line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }} }},
                    text: validData.map(function(d) {{ return makeHover(d, metric); }}),
                    hoverinfo: 'text', showlegend: true, _uniqueX: uniqueX,
                    customdata: validData
                }});
            }}
        }});
    }} else {{
        var traceType = (plotType === 'box') ? 'box' : 'violin';

        metrics.forEach(function(metric, mi) {{
            var color = getSeriesColor(mi);
            var groups = {{}};
            filtered.forEach(function(d) {{
                var key = String(d[xParam]);
                if (!groups[key]) groups[key] = [];
                if (d[metric] != null) groups[key].push(d[metric]);
            }});

            if (nMetrics === 1) {{
                // ONE trace spanning all categories (not one-per-key). A single box
                // trace sizes each box to the full category width; per-key traces
                // made them skinny and cluttered the legend with every combo.
                var col = getSeriesColor(0);
                if (traceType === 'box' && quartileMethod === 'custom') {{
                    // Custom-percentile box: precomputed stats, one entry per box.
                    var cx = [], cq1 = [], cmed = [], cq3 = [], clf = [], cuf = [], cmn = [];
                    uniqueX.forEach(function(key) {{
                        var vals = groups[key] || [];
                        if (!vals.length) return;
                        var bs = customBoxStats(vals);
                        cx.push(key); cq1.push(bs.q1); cmed.push(bs.median); cq3.push(bs.q3);
                        clf.push(bs.lowerfence); cuf.push(bs.upperfence); cmn.push(bs.mean);
                    }});
                    var tc = {{
                        x: cx, type: 'box', name: metric,
                        q1: cq1, median: cmed, q3: cq3, lowerfence: clf, upperfence: cuf,
                        marker: {{ color: col }}, line: {{ color: col }}
                    }};
                    if (showBoxmean) {{ tc.mean = cmn; tc.boxmean = 'sd'; }}
                    traces.push(tc);
                }} else {{
                    var xarr = [], yarr = [];
                    uniqueX.forEach(function(key) {{
                        (groups[key] || []).forEach(function(v) {{ xarr.push(key); yarr.push(v); }});
                    }});
                    var t = {{
                        x: xarr, y: yarr, type: traceType, name: metric,
                        marker: {{ color: col, size: 3, opacity: 0.5 }},
                    }};
                    if (traceType === 'box') {{
                        t.boxmean = showBoxmean ? 'sd' : false;
                        t.quartilemethod = effectiveQuartileMethod();
                    }}
                    if (traceType === 'violin') {{
                        t.box = {{ visible: true }};
                        t.meanline = {{ visible: true }};
                        t.line = {{ color: col }};
                        t.quartilemethod = effectiveQuartileMethod();
                    }}
                    traces.push(t);
                }}
            }} else {{
                uniqueX.forEach(function(key) {{
                    var vals = groups[key] || [];
                    var t = {{
                        x: vals.map(function() {{ return key; }}),
                        y: vals, type: traceType,
                        name: metric, legendgroup: metric,
                        showlegend: (key === uniqueX[0]),
                        marker: {{ color: color, size: 3, opacity: 0.5 }},
                    }};
                    if (traceType === 'box') {{
                        t.boxmean = showBoxmean ? 'sd' : false;
                        t.quartilemethod = effectiveQuartileMethod();
                    }}
                    if (traceType === 'violin') {{
                        t.box = {{ visible: true }};
                        t.meanline = {{ visible: true }};
                        t.line = {{ color: color }};
                        t.scalegroup = metric + '_' + key;
                        t.quartilemethod = effectiveQuartileMethod();
                    }}
                    traces.push(t);
                }});
            }}
        }});
    }}
    return traces;
}}

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
        var allXRaw = filtered.map(function(d) {{ return String(d[xParam]); }});
        var uniqueX = [...new Set(allXRaw)];
        if (xParam !== 'activation' && xParam !== 'combo') {{
            uniqueX.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
        }} else {{ uniqueX.sort(); }}
        var xMap = {{}};
        uniqueX.forEach(function(v, i) {{ xMap[v] = i; }});

        if (showOffset) {{
            var spread = 0.6;
            var step = nGroups > 1 ? spread / (nGroups - 1) : 0;
            var startOffset = nGroups > 1 ? -spread / 2 : 0;

            keys.forEach(function(key, ci) {{
                var sub = groups[key];
                var offset = startOffset + ci * step;
                var xNums = sub.map(function(d) {{
                    return xMap[String(d[xParam])] + offset + (Math.random() - 0.5) * 0.08;
                }});
                traces.push({{
                    x: xNums, y: sub.map(function(d) {{ return d[metric]; }}),
                    mode: 'markers', type: 'scatter', name: key, showlegend: true,
                    marker: {{ size: 6, color: colorPalette[ci % colorPalette.length], opacity: 0.6,
                               line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }} }},
                    text: sub.map(function(d) {{ return makeHover(d, metric); }}),
                    hoverinfo: 'text', _uniqueX: uniqueX,
                    customdata: sub
                }});
            }});
        }} else {{
            keys.forEach(function(key, ci) {{
                var sub = groups[key];
                traces.push({{
                    x: sub.map(function(d) {{ return String(d[xParam]); }}),
                    y: sub.map(function(d) {{ return d[metric]; }}),
                    mode: 'markers', type: 'scatter', name: key, showlegend: true,
                    marker: {{ size: 6, color: colorPalette[ci % colorPalette.length], opacity: 0.6,
                               line: {{ width: 0.5, color: 'rgba(0,0,0,0.15)' }} }},
                    text: sub.map(function(d) {{ return makeHover(d, metric); }}),
                    hoverinfo: 'text',
                    customdata: sub
                }});
            }});
        }}
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
            if (traceType === 'box') {{
                t.boxmean = showBoxmean ? 'sd' : false;
                t.quartilemethod = effectiveQuartileMethod();
            }}
            if (traceType === 'violin') {{
                t.box = {{ visible: true }};
                t.meanline = {{ visible: true }};
                t.line = {{ color: colorPalette[ci % colorPalette.length] }};
                t.quartilemethod = effectiveQuartileMethod();
            }}
            traces.push(t);
        }});
    }}
    return traces;
}}

// --- Main rebuild ---
function rebuild() {{
    var xParam = document.getElementById('xAxis').value;
    var yLabel = document.getElementById('yLabel').value;
    var colorBy = document.getElementById('colorBy').value;
    var metrics = dataSeries.map(function(s) {{ return s.metric; }});
    var multiData = metrics.length > 1;

    // v16: filters apply to the raw reps FIRST, then (optionally) collapse to
    // one record per combo. So Avg/Avg+Spread respect every active filter.
    var paramFiltered = applyFilter();
    // Recurrence is measured over the current-filter set (BEFORE the min-cycles
    // cull, so the cull can't shrink the very counts it filters on). Then hide
    // architectures that don't span at least `minCycles` cycles.
    computeArchRecurrence(paramFiltered);
    var minCycles = parseInt(document.getElementById('minCycles').value, 10) || 1;
    if (minCycles > 1) {{
        paramFiltered = paramFiltered.filter(function(d) {{ return archRec(d).c >= minCycles; }});
    }}
    // Annotate the raw reps with recurrence first (used by the Avg+Spread rep
    // overlay grouping when Color-by is a recurrence field).
    paramFiltered.forEach(function(d) {{
        var r = archRec(d);
        d.cyc_recurrence = r.c; d.lt_recurrence = r.l;
    }});
    var records = (showMode === 'raw') ? paramFiltered : aggregateRecords(paramFiltered);
    var filtered = records.filter(function(d) {{ return d[metrics[0]] != null; }});
    // Aggregated records are fresh objects, so annotate them too (trace builders
    // group on d[colorBy], and hover/labels read these).
    filtered.forEach(function(d) {{
        var r = archRec(d);
        d.cyc_recurrence = r.c; d.lt_recurrence = r.l;
    }});
    var missingCount = records.length - filtered.length;

    var unit = (showMode === 'raw') ? 'Models' : 'Combos';
    document.getElementById('modelCount').textContent = unit + ' (' + filtered.length + ')';
    var missingEl = document.getElementById('missingCount');
    if (missingCount > 0) {{
        missingEl.textContent = 'Missing ' + metrics[0] + ' (' + missingCount + ')';
        missingEl.style.display = '';
    }} else {{
        missingEl.style.display = 'none';
    }}

    // Guardrail (item 5/B): per-combo box/violin is only readable when filtered
    // down. Warn loudly rather than silently drawing a wall of boxes.
    var comboWarnEl = document.getElementById('comboWarn');
    var nUniqueX = new Set(filtered.map(function(d) {{ return String(d[xParam]); }})).size;
    if ((plotType === 'box' || plotType === 'violin') && xParam === 'combo' && nUniqueX > 120) {{
        comboWarnEl.textContent = nUniqueX + ' combos - filter to a leadtime for readable boxes';
        comboWarnEl.style.display = '';
    }} else {{
        comboWarnEl.style.display = 'none';
    }}

    var traces;
    var useNumericX = false;
    var legendTitle = '';

    if (showMode === 'avgspread' && plotType === 'scatter') {{
        // Mean dots + layered nested fences (categorical X; offset still ignored).
        // color-by (when not 'none') splits into one legend-toggleable group per
        // value so bad combos can be culled here too.
        var avgColorBy = multiData ? 'none' : colorBy;
        // paramFiltered = the raw per-rep rows (pre-aggregation); passed so the
        // "Reps" toggle can overlay individual model dots on the mean+spread.
        traces = buildAvgSpreadScatter(filtered, metrics[0], xParam, avgColorBy, paramFiltered, showReps);
        useNumericX = false;
        legendTitle = (avgColorBy && avgColorBy !== 'none') ? (xAxisLabels[avgColorBy] || '') : 'Spread';
    }} else if (multiData || colorBy === 'none') {{
        // Note: colorBy === xParam is intentionally NOT short-circuited here.
        // Coloring by the same field as X (e.g. X=Combo, Color=Combo) gives one
        // clickable legend entry per value — the intended cull workflow: put
        // combos on X, then hide bad ones via the legend until only finalists remain.
        traces = buildMultiSeriesTraces(filtered, metrics, xParam);
        useNumericX = (multiData || showOffset) && plotType === 'scatter';
        legendTitle = multiData ? 'Metric' : '';
    }} else {{
        traces = buildColorByTraces(filtered, metrics[0], xParam, colorBy);
        useNumericX = showOffset && plotType === 'scatter';
        legendTitle = xAxisLabels[colorBy] || '';
    }}

    var allXRaw = filtered.map(function(d) {{ return String(d[xParam]); }});
    var uniqueX = [...new Set(allXRaw)];
    if (xParam !== 'activation' && xParam !== 'combo') {{
        uniqueX.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
    }} else {{ uniqueX.sort(); }}

    var xAxisConfig;
    if (plotType === 'scatter' && useNumericX) {{
        xAxisConfig = {{
            title: xAxisLabels[xParam], automargin: true,
            tickvals: uniqueX.map(function(v, i) {{ return i; }}),
            ticktext: uniqueX, zeroline: false
        }};
    }} else {{
        xAxisConfig = {{
            title: xAxisLabels[xParam], type: 'category', automargin: true,
            categoryorder: 'array', categoryarray: uniqueX
        }};
    }}
    // Long combo labels: angle them and let automargin reserve room so they don't
    // overlap the axis title or get cut off.
    if (xParam === 'combo') xAxisConfig.tickangle = -45;

    var yaxis = {{ title: yLabel, rangemode: (plotType === 'scatter') ? 'normal' : 'tozero' }};
    if (pinY && pinYRange) {{
        yaxis.range = pinYRange.slice();
        yaxis.autorange = false;
    }}

    var h = Math.max(500, window.innerHeight - 180);
    var layout = {{
        xaxis: xAxisConfig,
        yaxis: yaxis,
        height: h, margin: {{ t: 30, b: 60, l: 70, r: 30 }},
        boxmode: showOffset ? 'group' : 'overlay',
        violinmode: showOffset ? 'group' : 'overlay',
        hovermode: 'closest',
        showlegend: showLegend,
        legend: {{ title: {{ text: legendTitle }} }},
        dragmode: 'select'
    }};
    // Dark mode: dark paper/plot backgrounds + light font/grid so the chart
    // matches the dark toolbar. Axis line/grid colors nudged for contrast.
    if (darkMode) {{
        layout.paper_bgcolor = '#1e1f22';
        layout.plot_bgcolor = '#1e1f22';
        layout.font = {{ color: '#e3e3e3' }};
        xAxisConfig.gridcolor = '#3a3d42'; xAxisConfig.zerolinecolor = '#3a3d42';
        xAxisConfig.linecolor = '#555';
        yaxis.gridcolor = '#3a3d42'; yaxis.zerolinecolor = '#3a3d42';
        yaxis.linecolor = '#555';
    }}

    Plotly.newPlot('plotDiv', traces, layout, {{ responsive: true }});

    // Listen for box/lasso select events on scatter
    var plotDiv = document.getElementById('plotDiv');
    plotDiv.on('plotly_selected', function(eventData) {{
        selectedPoints = [];
        if (eventData && eventData.points) {{
            eventData.points.forEach(function(pt) {{
                if (pt.customdata) {{
                    selectedPoints.push(pt.customdata);
                }}
            }});
        }}
        updateTsButton();
    }});
    plotDiv.on('plotly_deselect', function() {{
        selectedPoints = [];
        updateTsButton();
    }});

    // Idea 1: click a dot/box on the graph to hide its combo directly (no legend
    // hunting). Only when Color-by is active — then each combo is its own trace (or
    // legendgroup), so hiding is unambiguous. Pooled mode is a single trace, where a
    // click would nuke everything, so it's disabled there. Item B then drops the slot.
    plotDiv.on('plotly_click', function(ev) {{
        if (!ev || !ev.points || !ev.points.length) return;
        if (document.getElementById('colorBy').value === 'none') return;
        var gd = document.getElementById('plotDiv');
        var ci = ev.points[0].curveNumber;
        var target = (gd.data[ci] || {{}}).legendgroup || (gd.data[ci] || {{}}).name;
        if (target == null) return;
        var idxs = [];
        (gd.data || []).forEach(function(tr, i) {{
            var g = (tr.legendgroup != null) ? tr.legendgroup : tr.name;
            if (g === target) idxs.push(i);
        }});
        if (idxs.length) Plotly.restyle(gd, {{ visible: 'legendonly' }}, idxs);
    }});

    // Item B: when a series is hidden via the legend, also drop its now-empty
    // x-axis category so the remaining values close ranks (categorical X only).
    // Recompute categoryarray from still-visible traces, then relayout. The
    // _syncingCats guard prevents the relayout from re-entering this handler.
    var _syncingCats = false;
    plotDiv.on('plotly_restyle', function() {{
        if (_syncingCats) return;
        var gd = document.getElementById('plotDiv');
        if (!gd.layout || !gd.layout.xaxis || gd.layout.xaxis.type !== 'category') return;
        var vis = {{}};
        (gd.data || []).forEach(function(tr) {{
            if (tr.visible === 'legendonly' || tr.visible === false) return;
            (tr.x || []).forEach(function(xv) {{ vis[String(xv)] = true; }});
        }});
        var cats = Object.keys(vis);
        if (xParam !== 'activation' && xParam !== 'combo') {{
            cats.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
        }} else {{ cats.sort(); }}
        _syncingCats = true;
        Plotly.relayout(gd, {{ 'xaxis.categoryarray': cats }})
            .then(function() {{ _syncingCats = false; }})
            .catch(function() {{ _syncingCats = false; }});
    }});
}}

// --- Init ---
initMultiSelect('msAct', 'activation');
initMultiSelect('msLayers', 'num_layers');
initMultiSelect('msNeurons', 'neurons');
initMultiSelect('msLT', 'leadtime');
initMultiSelect('msCy', 'cycle');
{dropout_init_js}

// Populate the Min-cycles dropdown 1..(total distinct cycles in the data).
(function() {{
    var cyc = {{}};
    allData.forEach(function(d) {{ cyc[d.cycle] = 1; }});
    var nCyc = Object.keys(cyc).length || 1;
    var sel = document.getElementById('minCycles');
    sel.innerHTML = '';
    for (var i = 1; i <= nCyc; i++) {{
        var o = document.createElement('option');
        o.value = String(i); o.textContent = String(i);
        sel.appendChild(o);
    }}
}})();

// Guardrail #1: if the data is already 1-row-per-combo, aggregation is a no-op.
// Disable the Avg / Avg+Spread buttons and say why (loud, not silent).
if (!aggEnabled) {{
    ['btnAvg', 'btnAvgSpread'].forEach(function(id) {{
        var b = document.getElementById(id);
        b.disabled = true;
        b.title = 'Needs per-rep data - only 1 row per combo detected ' +
                  '(looks like an already-averaged file). Feed the reps CSV to enable.';
    }});
}}

dataSeriesCounter = 1;
dataSeries = [{{ id: 1, metric: metricCols[0] }}];
renderDataSeries();

window.addEventListener('resize', function() {{
    rebuild();
    if (document.getElementById('tourBackdrop').style.display === 'block') tourShow();
}});
rebuild();
updateBasketCount();
</script>
</body></html>"""

    # Inline plotly.js so the emailed file is fully self-contained (works offline,
    # no CDN dependency). Done via str.replace (not the f-string) because the JS
    # source is full of literal braces that would break f-string parsing. Falls
    # back to the CDN tag if the offline bundle isn't available.
    try:
        import plotly.offline as _po
        _plotly_tag = f'<script type="text/javascript">{_po.get_plotlyjs()}</script>'
    except Exception as _e:
        print(f"  WARN: could not inline plotly.js ({_e}); falling back to CDN.")
        _plotly_tag = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    html = html.replace('<!--PLOTLY_JS_PLACEHOLDER-->', _plotly_tag)

    if output_dir:
        with open(f"{output_dir}/01_explorer.html", 'w', encoding='utf-8') as f:
            f.write(html)
        print("Saved: 01_explorer.html (v17)")
    return html
