'''
histogram plot visualizer

Date: 07/21/2026
Author: Ayesha Khan

Interactive histogram of tuning results. The X variable is chosen live in the
browser from a dropdown, so a single generated HTML lets you histogram any
metric (val_mae, val_mape, loss_value, ...) OR any hyperparameter (neurons,
num_layers, leadtime, cycle, activation) without regenerating.

Numeric columns render as binned histograms (bin count adjustable); categorical
columns render as per-category counts. An optional "Group by" dropdown splits
the distribution by a hyperparameter (overlaid for numeric, grouped bars for
categorical).

Follows the client-side data+JS convention of iteration_comparison_plot_v8:
all records are dumped once as JSON and a single rebuild() function does the
binning/aggregation in-browser.
'''

from typing import Optional
import json as _json
import pandas as pd

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Columns offered in the "X variable" dropdown, in preference order. Only those
# actually present in the data are kept.
NUMERIC_CANDIDATES = [
    'val_mae', 'val_mae12', 'mae_2021', 'mae12_2021',
    'val_mape', 'val_mse', 'val_loss', 'loss_value',
    'neurons', 'num_layers', 'leadtime', 'cycle',
]
CATEGORICAL_CANDIDATES = ['activation', 'metric_type']

# Columns offered in the "Group by" dropdown (discrete splits).
GROUP_CANDIDATES = ['activation', 'metric_type', 'leadtime', 'cycle']


def plot_histogram_v1(data: pd.DataFrame, metric_column: str,
                      output_dir: Optional[str] = None) -> Optional[str]:
    '''
    Build an interactive histogram explorer.

    Args:
        data: parsed tuning results (from TuningResultsLoader).
        metric_column: default column to histogram on first load.
        output_dir: if given, write 07_histogram.html here.

    Returns:
        The HTML string, or None if plotly is unavailable.
    '''
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.copy()

    numeric_cols = [c for c in NUMERIC_CANDIDATES if c in clean.columns]
    categorical_cols = [c for c in CATEGORICAL_CANDIDATES if c in clean.columns]
    x_cols = numeric_cols + categorical_cols
    group_cols = [c for c in GROUP_CANDIDATES if c in clean.columns]

    if not x_cols:
        print("Histogram: no plottable columns found. Skipping.")
        return None

    # One record per config row, carrying every candidate column. NaN -> None so
    # the JS filter can drop missing values.
    records = []
    for _, row in clean.iterrows():
        rec = {}
        for col in x_cols:
            val = row[col]
            if pd.isna(val):
                rec[col] = None
            elif col in categorical_cols:
                rec[col] = str(val)
            else:
                rec[col] = round(float(val), 6)
        records.append(rec)

    col_types = {c: 'numeric' for c in numeric_cols}
    col_types.update({c: 'categorical' for c in categorical_cols})

    default_x = metric_column if metric_column in x_cols else x_cols[0]

    x_options = "".join(
        f'<option value="{c}"{" selected" if c == default_x else ""}>{c}</option>\n'
        for c in x_cols
    )
    group_options = '<option value="" selected>None</option>\n' + "".join(
        f'<option value="{c}">{c}</option>\n' for c in group_cols
    )

    records_json = _json.dumps(records)
    col_types_json = _json.dumps(col_types)

    html = f"""<!DOCTYPE html>
<html><head>
<title>Histogram Explorer: {metric_column}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           margin: 0; padding: 20px; }}
    .controls {{ background: #f8f9fa; padding: 12px 20px; border-radius: 8px; margin-bottom: 8px; }}
    .controls h2 {{ margin: 0 0 8px 0; font-size: 18px; }}
    .controls p {{ margin: 4px 0 8px 0; font-size: 13px; color: #666; }}
    .filter-bar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 14px; }}
    .filter-group {{ display: flex; align-items: center; gap: 6px; }}
    .filter-group label {{ font-size: 13px; font-weight: 600; color: #555; white-space: nowrap; }}
    .filter-group input {{ font-size: 13px; padding: 4px 8px; border: 1px solid #ccc;
                           border-radius: 4px; width: 80px; }}
    .filter-group select {{ font-size: 13px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; }}
    .summary {{ background: #e7f3f4; padding: 10px 16px; border-radius: 6px; margin: 10px 0;
                font-size: 13px; border: 1px solid #52a8ad; }}
    #plotDiv {{ width: 100%; }}
</style>
</head><body>
<div class="controls">
    <h2>Histogram Explorer: {metric_column}</h2>
    <p>Pick any column to histogram. Numeric columns bin into a distribution;
       categorical columns show per-category counts. Optionally split by a hyperparameter.</p>
    <div class="filter-bar">
        <div class="filter-group">
            <label>X variable:</label>
            <select id="xVar" onchange="rebuild()">{x_options}</select>
        </div>
        <div class="filter-group">
            <label>Group by:</label>
            <select id="groupBy" onchange="rebuild()">{group_options}</select>
        </div>
        <div class="filter-group">
            <label>Bins:</label>
            <input type="number" id="binCount" value="30" min="2" max="200" step="1" onchange="rebuild()">
        </div>
    </div>
</div>
<div id="summary" class="summary"></div>
<div id="plotDiv"></div>

<script>
var allRecords = {records_json};
var colTypes = {col_types_json};

function makeTrace(vals, type, nbins, name) {{
    if (type === 'numeric') {{
        return {{ x: vals, type: 'histogram', nbinsx: nbins, name: name, opacity: 0.65 }};
    }}
    // categorical: count occurrences per category
    var counts = {{}};
    vals.forEach(function(v) {{ counts[v] = (counts[v] || 0) + 1; }});
    var cats = Object.keys(counts).sort();
    return {{
        x: cats,
        y: cats.map(function(c) {{ return counts[c]; }}),
        type: 'bar', name: name, opacity: 0.75,
    }};
}}

function rebuild() {{
    var xVar = document.getElementById('xVar').value;
    var groupBy = document.getElementById('groupBy').value;
    var nbins = parseInt(document.getElementById('binCount').value) || 30;
    var type = colTypes[xVar];

    var traces = [];
    var totalNonNull = 0;

    if (groupBy) {{
        var groups = {{}};
        allRecords.forEach(function(r) {{
            var g = r[groupBy];
            if (g === null || g === undefined) return;
            if (r[xVar] === null || r[xVar] === undefined) return;
            if (!(g in groups)) groups[g] = [];
            groups[g].push(r[xVar]);
        }});
        Object.keys(groups).sort().forEach(function(g) {{
            totalNonNull += groups[g].length;
            traces.push(makeTrace(groups[g], type, nbins, String(g)));
        }});
    }} else {{
        var vals = allRecords
            .map(function(r) {{ return r[xVar]; }})
            .filter(function(v) {{ return v !== null && v !== undefined; }});
        totalNonNull = vals.length;
        traces.push(makeTrace(vals, type, nbins, xVar));
    }}

    document.getElementById('summary').innerHTML =
        '<b>Summary:</b> ' + allRecords.length + ' total configs | ' +
        totalNonNull + ' non-null ' + xVar + ' values | ' +
        (type === 'numeric' ? nbins + ' bins' : 'categorical') +
        (groupBy ? ' | grouped by ' + groupBy + ' (' + traces.length + ' series)' : '');

    var layout = {{
        barmode: (type === 'numeric') ? 'overlay' : 'group',
        xaxis: {{ title: xVar, type: (type === 'categorical') ? 'category' : 'linear' }},
        yaxis: {{ title: 'Count' }},
        height: 600,
        margin: {{ t: 30, b: 70, l: 70, r: 30 }},
        showlegend: !!groupBy,
    }};

    Plotly.newPlot('plotDiv', traces, layout, {{ responsive: true }});
}}

rebuild();
</script>
</body></html>"""

    if output_dir:
        path = f"{output_dir}/07_histogram.html"
        with open(path, 'w') as f:
            f.write(html)
        print("Saved: 07_histogram.html")

    return html
