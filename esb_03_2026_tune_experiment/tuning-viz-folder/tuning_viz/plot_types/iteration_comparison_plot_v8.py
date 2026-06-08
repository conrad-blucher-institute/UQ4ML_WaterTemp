"""
V8 Iteration Comparison: per-combo stability analysis.
Groups by hyperparameter combo, shows mean + std across iterations.
Manual failure threshold input. Flags high-variance combos.
Only useful with 2+ iterations.
"""

from typing import Optional
import json as _json
import pandas as pd
import numpy as np

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_iteration_comparison_v8(data: pd.DataFrame, metric_column: str,
                                   output_dir: Optional[str] = None) -> Optional[str]:
    """
    Per-combo stability analysis with adjustable failure threshold.
    Shows mean, std, min, max across iterations for each hyperparameter combo.
    User can set failure threshold via input field — combos flagged accordingly.
    """
    if 'iteration' not in data.columns or data['iteration'].nunique() < 2:
        print("Only 1 iteration in data. Skipping iteration comparison.")
        print("To test, use test_06.py to simulate multiple iterations.")
        return None

    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()

    # Define combo key
    clean['combo'] = (
        clean['activation'] + '_L' + clean['num_layers'].astype(int).astype(str)
        + '_N' + clean['neurons'].astype(int).astype(str)
        + '_LT' + clean['leadtime'].astype(int).astype(str) + 'h'
        + '_C' + clean['cycle'].astype(int).astype(str)
    )

    # Compute stats per combo
    stats = clean.groupby('combo')[metric_column].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    stats.columns = ['combo', 'mean', 'std', 'min', 'max', 'n_iters']
    stats['std'] = stats['std'].fillna(0)
    stats = stats.sort_values('mean').reset_index(drop=True)

    records = []
    for _, row in stats.iterrows():
        records.append({
            'combo': row['combo'],
            'mean': round(float(row['mean']), 4),
            'std': round(float(row['std']), 4),
            'min': round(float(row['min']), 4),
            'max': round(float(row['max']), 4),
            'n_iters': int(row['n_iters']),
        })

    data_json = _json.dumps(records)
    default_threshold = round(float(stats['mean'].mean() + 2 * stats['mean'].std()), 2)

    html = f"""<!DOCTYPE html>
<html><head>
<title>Iteration Stability: {metric_column}</title>
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
    .btn {{ font-size: 12px; padding: 5px 12px; border: 1px solid #ccc; border-radius: 4px;
            background: white; cursor: pointer; }}
    .btn:hover {{ background: #e9ecef; }}
    .summary {{ background: #fff3cd; padding: 10px 16px; border-radius: 6px; margin: 10px 0;
                font-size: 13px; border: 1px solid #ffc107; }}
    #plotDiv {{ width: 100%; }}
</style>
</head><body>
<div class="controls">
    <h2>Per-Combo Stability Analysis: {metric_column}</h2>
    <p>Each bar = one hyperparameter combo. Error bars show ± 1 std across iterations.
       Red bars exceed the failure threshold. High std = unstable combo.</p>
    <div class="filter-bar">
        <div class="filter-group">
            <label>Failure threshold:</label>
            <input type="number" id="threshold" value="{default_threshold}" step="0.1" onchange="rebuild()">
        </div>
        <div class="filter-group">
            <label>Sort by:</label>
            <select id="sortBy" onchange="rebuild()">
                <option value="mean" selected>Mean</option>
                <option value="std">Std (variance)</option>
                <option value="max">Max</option>
            </select>
        </div>
        <div class="filter-group">
            <label>Show:</label>
            <select id="showFilter" onchange="rebuild()">
                <option value="all" selected>All combos</option>
                <option value="good">Below threshold</option>
                <option value="bad">Above threshold</option>
                <option value="unstable">High variance (std &gt; 1)</option>
            </select>
        </div>
        <div class="filter-group">
            <label>Top N:</label>
            <input type="number" id="topN" value="50" min="5" max="500" step="5" onchange="rebuild()">
        </div>
    </div>
</div>
<div id="summary" class="summary"></div>
<div id="plotDiv"></div>

<script>
var allStats = {data_json};
var metricCol = "{metric_column}";

function rebuild() {{
    var threshold = parseFloat(document.getElementById('threshold').value);
    var sortBy = document.getElementById('sortBy').value;
    var showFilter = document.getElementById('showFilter').value;
    var topN = parseInt(document.getElementById('topN').value);

    // Filter
    var filtered = allStats.filter(function(d) {{
        if (showFilter === 'good') return d.mean < threshold;
        if (showFilter === 'bad') return d.mean >= threshold;
        if (showFilter === 'unstable') return d.std > 1.0;
        return true;
    }});

    // Sort
    filtered.sort(function(a, b) {{ return a[sortBy] - b[sortBy]; }});

    // Limit
    filtered = filtered.slice(0, topN);

    // Summary
    var nAbove = allStats.filter(function(d) {{ return d.mean >= threshold; }}).length;
    var nUnstable = allStats.filter(function(d) {{ return d.std > 1.0; }}).length;
    document.getElementById('summary').innerHTML =
        '<b>Summary:</b> ' + allStats.length + ' total combos | ' +
        nAbove + ' above threshold (' + threshold + ') | ' +
        nUnstable + ' high variance (std > 1.0) | Showing ' + filtered.length + ' combos';

    // Colors
    var barColors = filtered.map(function(d) {{
        return d.mean >= threshold ? 'rgba(220, 50, 50, 0.7)' : 'rgba(70, 130, 180, 0.7)';
    }});

    var trace = {{
        x: filtered.map(function(d) {{ return d.combo; }}),
        y: filtered.map(function(d) {{ return d.mean; }}),
        error_y: {{
            type: 'data',
            array: filtered.map(function(d) {{ return d.std; }}),
            visible: true,
            color: 'rgba(0,0,0,0.3)'
        }},
        type: 'bar',
        marker: {{ color: barColors }},
        text: filtered.map(function(d) {{
            return 'Mean: ' + d.mean + '<br>Std: ' + d.std +
                   '<br>Min: ' + d.min + '<br>Max: ' + d.max +
                   '<br>Iterations: ' + d.n_iters;
        }}),
        hoverinfo: 'text+x',
    }};

    var layout = {{
        xaxis: {{
            title: 'Hyperparameter Combo',
            tickangle: 45,
            tickfont: {{ size: 8 }},
            type: 'category'
        }},
        yaxis: {{ title: metricCol + ' (mean ± std)' }},
        height: 600,
        margin: {{ t: 30, b: 180, l: 70, r: 30 }},
        shapes: [{{
            type: 'line',
            x0: -0.5, x1: filtered.length - 0.5,
            y0: threshold, y1: threshold,
            line: {{ color: 'red', width: 2, dash: 'dash' }}
        }}],
        annotations: [{{
            x: 0, y: threshold,
            xref: 'paper', yref: 'y',
            text: 'Failure threshold: ' + threshold,
            showarrow: false,
            font: {{ color: 'red', size: 11 }},
            xanchor: 'left',
            yanchor: 'bottom'
        }}]
    }};

    Plotly.newPlot('plotDiv', [trace], layout, {{ responsive: true }});
}}

rebuild();
</script>
</body></html>"""

    if output_dir:
        path = f"{output_dir}/06_iteration_stability.html"
        with open(path, 'w') as f:
            f.write(html)
        print("Saved: 06_iteration_stability.html")

    return html
