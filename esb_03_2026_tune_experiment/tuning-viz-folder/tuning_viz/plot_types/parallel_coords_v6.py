"""
V6 Parallel Coordinates: single combined plot, tick marks = actual values only.
Height auto-resizes to fill viewport.
"""

from typing import Optional
import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_parallel_coords_v6(data: pd.DataFrame, metric_column: str,
                              output_dir: Optional[str] = None) -> Optional[str]:
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()

    activations = sorted(clean['activation'].unique())
    act_map = {act: i for i, act in enumerate(activations)}
    clean['activation_code'] = clean['activation'].map(act_map)

    unique_layers = sorted(clean['num_layers'].unique())
    unique_neurons = sorted(clean['neurons'].unique())
    unique_leadtimes = sorted(clean['leadtime'].unique())
    unique_cycles = sorted(clean['cycle'].unique())

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=clean[metric_column],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title=metric_column),
            ),
            dimensions=[
                dict(
                    label='Activation',
                    values=clean['activation_code'],
                    tickvals=list(act_map.values()),
                    ticktext=list(act_map.keys()),
                ),
                dict(
                    label='Num Layers',
                    values=clean['num_layers'],
                    tickvals=[int(v) for v in unique_layers],
                    ticktext=[str(int(v)) for v in unique_layers],
                ),
                dict(
                    label='Hidden Units',
                    values=clean['neurons'],
                    tickvals=[int(v) for v in unique_neurons],
                    ticktext=[str(int(v)) for v in unique_neurons],
                ),
                dict(
                    label='Leadtime (h)',
                    values=clean['leadtime'],
                    tickvals=[int(v) for v in unique_leadtimes],
                    ticktext=[str(int(v)) for v in unique_leadtimes],
                ),
                dict(
                    label='Cycle',
                    values=clean['cycle'],
                    tickvals=[int(v) for v in unique_cycles],
                    ticktext=[str(int(v)) for v in unique_cycles],
                ),
                dict(
                    label=metric_column,
                    values=clean[metric_column],
                ),
            ],
        )
    )

    fig.update_layout(
        title=f"Parallel Coordinates: {metric_column} (All Data)",
        margin=dict(l=100, r=50, t=80, b=30),
    )

    # Custom HTML to make height fill viewport
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    html = f"""<!DOCTYPE html>
<html><head>
<title>Parallel Coordinates: {metric_column}</title>
<style>
    html, body {{ margin: 0; padding: 0; height: 100%; overflow: hidden; }}
    #plotWrap {{ width: 100%; height: 100vh; }}
</style>
</head><body>
<div id="plotWrap">{fig_html}</div>
<script>
function resizePlot() {{
    var gd = document.querySelector('.plotly-graph-div');
    if (gd) {{
        var h = window.innerHeight;
        Plotly.relayout(gd, {{ height: h, width: null }});
    }}
}}
window.addEventListener('resize', resizePlot);
window.addEventListener('load', function() {{ setTimeout(resizePlot, 200); }});
</script>
</body></html>"""

    if output_dir:
        path = f"{output_dir}/05_parallel_coords.html"
        with open(path, 'w') as f:
            f.write(html)
        print("Saved: 05_parallel_coords.html")

    return html
