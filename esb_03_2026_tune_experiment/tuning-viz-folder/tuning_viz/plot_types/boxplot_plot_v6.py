"""
V6 Boxplot: x-axis selector dropdown to choose grouping variable.
Optional color-by dropdown. Cycle dropdown. JS-based rebuild.
Replaces layers_impact since this now covers that use case.
"""

from typing import Optional
import json as _json
import pandas as pd

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_boxplot_v6(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> Optional[str]:
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

    params = ['activation', 'num_layers', 'neurons', 'leadtime', 'cycle']
    param_labels = {
        'activation': 'Activation Function',
        'num_layers': 'Number of Layers',
        'neurons': 'Hidden Units',
        'leadtime': 'Leadtime (h)',
        'cycle': 'Cycle',
    }

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
        <div style="border-left: 2px solid #ddd; height: 24px;"></div>
        <div class="filter-group">
            <label>Cycle:</label>
            <select id="fCycle" onchange="rebuild()">
                <option value="ALL" selected>All</option>
                {"".join(f'<option value="{int(c)}">{int(c)}</option>' for c in sorted(clean['cycle'].unique()))}
            </select>
        </div>
    </div>
</div>
<div id="plotDiv"></div>

<script>
var allData = {data_json};
var metricCol = "{metric_column}";
var paramLabels = {_json.dumps(param_labels)};

var colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
              '#a65628','#f781bf','#999999','#66c2a5','#fc8d62'];

function rebuild() {{
    var xParam = document.getElementById('xAxis').value;
    var colorBy = document.getElementById('colorBy').value;
    var fCycle = document.getElementById('fCycle').value;

    var filtered = allData.filter(function(d) {{
        if (fCycle !== 'ALL' && d.cycle !== parseInt(fCycle)) return false;
        return true;
    }});

    var traces = [];

    if (colorBy === 'none' || colorBy === xParam) {{
        // Single color boxplot grouped by x
        var groups = {{}};
        filtered.forEach(function(d) {{
            var key = String(d[xParam]);
            if (!groups[key]) groups[key] = [];
            groups[key].push(d.metric);
        }});

        var sortedKeys = Object.keys(groups);
        if (xParam !== 'activation') {{
            sortedKeys.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
        }} else {{
            sortedKeys.sort();
        }}

        sortedKeys.forEach(function(key) {{
            traces.push({{
                y: groups[key],
                type: 'box',
                name: key,
                boxmean: 'sd'
            }});
        }});
    }} else {{
        // Colored boxplot: group by x, sub-group by color param
        var colorVals = [...new Set(filtered.map(function(d) {{ return String(d[colorBy]); }}))];
        if (colorBy !== 'activation') {{
            colorVals.sort(function(a, b) {{ return parseFloat(a) - parseFloat(b); }});
        }} else {{
            colorVals.sort();
        }}

        colorVals.forEach(function(cv, ci) {{
            var sub = filtered.filter(function(d) {{ return String(d[colorBy]) === cv; }});
            var xVals = sub.map(function(d) {{ return String(d[xParam]); }});
            var yVals = sub.map(function(d) {{ return d.metric; }});

            traces.push({{
                x: xVals,
                y: yVals,
                type: 'box',
                name: cv,
                boxmean: 'sd',
                marker: {{ color: colors[ci % colors.length] }}
            }});
        }});
    }}

    var layout = {{
        xaxis: {{
            title: paramLabels[xParam],
            type: 'category'
        }},
        yaxis: {{ title: metricCol }},
        height: 550,
        margin: {{ t: 30, b: 60, l: 70, r: 30 }},
        boxmode: 'group',
        hovermode: 'closest'
    }};

    Plotly.newPlot('plotDiv', traces, layout, {{ responsive: true }});
}}

rebuild();
</script>
</body></html>"""

    if output_dir:
        path = f"{output_dir}/03_boxplot.html"
        with open(path, 'w') as f:
            f.write(html)
        print("Saved: 03_boxplot.html")

    return html
