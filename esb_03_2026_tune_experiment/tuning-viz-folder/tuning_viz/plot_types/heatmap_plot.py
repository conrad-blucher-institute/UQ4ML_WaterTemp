"""
Heatmap: activation functions vs neuron counts, colored by metric.
"""

from typing import Optional, Union, List
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_heatmap(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None, interactive: bool = True) -> Optional[Union[go.Figure, plt.Figure]]:
    """
    Heatmap: activation functions vs neuron counts, colored by average metric.
    Use best value across iterations for each combination.
    Generates one heatmap per leadtime.
    
    Args:
        data: Parsed tuning results DataFrame
        metric_column: Name of the metric column to visualize
        output_dir: Optional directory to save plots
        interactive: If True, use Plotly. If False, use matplotlib.
        
    Returns:
        Plotly figure object, matplotlib figure, or list of figures if multiple leadtimes
    """
    if interactive and not PLOTLY_AVAILABLE:
        print("Plotly not available. Falling back to matplotlib.")
        interactive = False
    
    # Focus on single cycle/leadtime for clarity (average across if multiple)
    leadtimes = sorted(data['leadtime'].unique())
    
    figures = []
    
    for leadtime in leadtimes:
        subset = data[data['leadtime'] == leadtime]
        
        # Pivot table: activation x neurons, values = mean metric
        pivot = subset.pivot_table(
            values=metric_column,
            index='activation',
            columns='neurons',
            aggfunc='min'  # Use best (min) value
        )
        
        if interactive:
            fig = go.Figure(
                data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale='RdYlGn_r',
                    text=np.round(pivot.values, 4),
                    texttemplate='%{text:.3f}',
                    textfont={"size": 10},
                    colorbar=dict(title=metric_column)
                )
            )
            fig.update_layout(
                title=f"Activation vs Neurons (Leadtime: {leadtime}h)",
                xaxis_title="Hidden Units",
                yaxis_title="Activation Function",
                height=400,
                width=900
            )
            
            if output_dir:
                fig.write_html(f"{output_dir}/02_heatmap_leadtime_{leadtime}h.html")
                print(f"Saved: 02_heatmap_leadtime_{leadtime}h.html")
            
            figures.append(fig)
        else:
            # Matplotlib version
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax)
            ax.set_title(f"Activation vs Neurons (Leadtime: {leadtime}h)")
            ax.set_xlabel("Hidden Units")
            ax.set_ylabel("Activation Function")
            
            if output_dir:
                fig.savefig(f"{output_dir}/02_heatmap_leadtime_{leadtime}h.png", dpi=150, bbox_inches='tight')
                print(f"Saved: 02_heatmap_leadtime_{leadtime}h.png")
            
            figures.append(fig)
    
    return figures[0] if len(figures) == 1 else figures
