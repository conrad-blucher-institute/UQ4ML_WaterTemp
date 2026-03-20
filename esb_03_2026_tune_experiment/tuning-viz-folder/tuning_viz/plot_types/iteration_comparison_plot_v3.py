"""
V3 Iteration Comparison: best metric per iteration.
Only generated when 2+ iterations exist.
"""

from typing import Optional, List
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_iteration_comparison_v3(data: pd.DataFrame, metric_column: str,
                                  output_dir: Optional[str] = None) -> List[go.Figure]:
    """
    Bar chart: best metric per iteration. Skips if only 1 iteration.
    """
    if 'iteration' not in data.columns or data['iteration'].nunique() < 2:
        print("Only 1 iteration in data. Skipping iteration comparison plot.")
        return []

    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return []

    best_per_iter = data.groupby('iteration')[metric_column].min()

    fig = go.Figure(
        data=go.Bar(
            x=[f"Iteration {i}" for i in best_per_iter.index],
            y=best_per_iter.values,
            text=[f"{v:.4f}" for v in best_per_iter.values],
            textposition='auto',
            marker=dict(color='steelblue'),
            name=f"Best {metric_column}",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=f"Best {metric_column} per Iteration",
        xaxis_title="Iteration",
        yaxis_title=f"Best {metric_column}",
        height=450,
        legend=dict(title="Metric"),
    )

    if output_dir:
        fig.write_html(f"{output_dir}/06_iteration_comparison.html")
        print("Saved: 06_iteration_comparison.html")

    return [fig]
