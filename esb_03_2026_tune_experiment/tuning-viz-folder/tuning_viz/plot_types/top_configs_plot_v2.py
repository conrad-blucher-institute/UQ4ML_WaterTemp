"""
V2 Top Configs: interactive table of top 50 configurations.
Columns: Rank, Hyperparameter Combo, metric value.
"""

from typing import Optional
import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_top_configs_v2(data: pd.DataFrame, metric_column: str, top_n: int = 50, output_dir: Optional[str] = None) -> Optional[go.Figure]:
    """
    Table showing top N configurations ranked by metric.
    Columns: Rank, Activation, Layers, Neurons, Leadtime, Cycle, metric value.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    top = data.nsmallest(top_n, metric_column).copy()
    top['rank'] = range(1, len(top) + 1)

    # Build the hyperparameter combo string
    top['combo'] = (
        top['activation'] + ' | L' + top['num_layers'].astype(int).astype(str)
        + ' | N' + top['neurons'].astype(int).astype(str)
        + ' | LT' + top['leadtime'].astype(int).astype(str) + 'h'
        + ' | C' + top['cycle'].astype(int).astype(str)
    )

    # Color rows by metric value (green = low/good, red = high/bad)
    metric_vals = top[metric_column].values
    norm = (metric_vals - metric_vals.min()) / (metric_vals.max() - metric_vals.min() + 1e-9)
    row_colors = [
        f"rgba({int(200 * n)}, {int(200 * (1 - n))}, 80, 0.25)"
        for n in norm
    ]

    fig = go.Figure(
        data=go.Table(
            header=dict(
                values=['<b>Rank</b>', '<b>Hyperparameter Combo</b>', f'<b>{metric_column}</b>'],
                fill_color='rgb(55, 55, 55)',
                font=dict(color='white', size=13),
                align='center',
                height=35,
            ),
            cells=dict(
                values=[
                    top['rank'].tolist(),
                    top['combo'].tolist(),
                    [f"{v:.4f}" for v in top[metric_column]],
                ],
                fill_color=[
                    ['white'] * len(top),
                    row_colors,
                    row_colors,
                ],
                font=dict(size=12),
                align=['center', 'left', 'center'],
                height=28,
            ),
        )
    )

    fig.update_layout(
        title=f"Top {top_n} Configurations — {metric_column}",
        height=60 + 35 + 28 * len(top),
    )

    if output_dir:
        fig.write_html(f"{output_dir}/04_top_configs_v2.html")
        print("Saved: 04_top_configs_v2.html")

    return fig
