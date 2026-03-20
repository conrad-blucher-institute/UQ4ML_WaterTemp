"""
V7 Parallel Coordinates: single combined plot, actual tick values only.
Height auto-resizes. Larger bold fonts for readability.
"""

from typing import Optional
import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_parallel_coords_v7(data: pd.DataFrame, metric_column: str,
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

    label_font = dict(size=15, color='black')
    tick_font = dict(size=13, color='#333')

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=clean[metric_column],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title=dict(text=metric_column, font=dict(size=14))),
            ),
            labelfont=label_font,
            tickfont=tick_font,
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
        title=dict(
            text=f"Parallel Coordinates: {metric_column} (All Data)",
            font=dict(size=18),
        ),
        margin=dict(l=100, r=50, t=80, b=30),
    )

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
    if (gd) Plotly.relayout(gd, {{ height: window.innerHeight, width: null }});
}}
window.addEventListener('resize', resizePlot);
window.addEventListener('load', function() {{ setTimeout(resizePlot, 200); }});
</script>
</body></html>"""

    if output_dir:
        with open(f"{output_dir}/05_parallel_coords.html", 'w') as f:
            f.write(html)
        print("Saved: 05_parallel_coords.html")
    return html
