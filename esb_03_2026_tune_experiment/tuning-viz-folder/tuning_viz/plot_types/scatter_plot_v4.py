"""
V4 Scatter plots: categorical x-axes, combined + per-cycle views.
Color = activation, symbol = leadtime.
Dropdown menus to filter by activation and leadtime independently.
"""

from typing import Optional, List
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def _build_scatter_figure(data: pd.DataFrame, metric_column: str, title_suffix: str = "") -> go.Figure:
    """Build a single scatter figure with dropdown filters for activation and leadtime."""
    clean = data.copy()
    clean['neurons_cat'] = clean['neurons'].astype(int).astype(str)
    clean['num_layers_cat'] = clean['num_layers'].astype(int).astype(str)
    clean['leadtime_label'] = clean['leadtime'].astype(int).astype(str) + 'h'

    x_axes = [
        ('neurons_cat', 'Hidden Units'),
        ('num_layers_cat', 'Number of Layers'),
    ]

    activations = sorted(clean['activation'].unique())
    leadtime_labels = sorted(clean['leadtime_label'].unique(), key=lambda x: int(x.replace('h', '')))

    act_colors = px.colors.qualitative.Set1
    act_color_map = {act: act_colors[i % len(act_colors)] for i, act in enumerate(activations)}

    symbol_list = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'star', 'hexagon']
    lt_symbol_map = {lt: symbol_list[i % len(symbol_list)] for i, lt in enumerate(leadtime_labels)}

    fig = make_subplots(
        rows=len(x_axes), cols=1,
        subplot_titles=[label for _, label in x_axes],
        vertical_spacing=0.12,
    )

    # Store metadata per trace for dropdown filtering
    trace_activations = []
    trace_leadtimes = []
    seen_legend = set()

    for row_idx, (x_col, x_label) in enumerate(x_axes, start=1):
        cat_order = sorted(clean[x_col].unique(), key=lambda x: int(x))

        for activation in activations:
            for lt_label in leadtime_labels:
                subset = clean[(clean['activation'] == activation) & (clean['leadtime_label'] == lt_label)]
                if subset.empty:
                    continue

                legend_name = f"{activation} | {lt_label}"
                show = legend_name not in seen_legend
                seen_legend.add(legend_name)

                trace_activations.append(activation)
                trace_leadtimes.append(lt_label)

                fig.add_trace(
                    go.Scatter(
                        x=subset[x_col],
                        y=subset[metric_column],
                        mode='markers',
                        name=legend_name,
                        legendgroup=legend_name,
                        showlegend=show,
                        marker=dict(
                            size=7,
                            color=act_color_map[activation],
                            symbol=lt_symbol_map[lt_label],
                            opacity=0.7,
                            line=dict(width=0.5, color='white'),
                        ),
                        text=[
                            f"Activation: {a}<br>Neurons: {n}<br>Layers: {l}<br>"
                            f"Leadtime: {lt}<br>{metric_column}: {m:.4f}"
                            for a, n, l, lt, m in zip(
                                subset['activation'], subset['neurons'],
                                subset['num_layers'], subset['leadtime_label'],
                                subset[metric_column],
                            )
                        ],
                        hoverinfo='text',
                    ),
                    row=row_idx, col=1,
                )

        fig.update_xaxes(
            title_text=x_label, row=row_idx, col=1,
            type='category',
            categoryorder='array',
            categoryarray=cat_order,
        )
        fig.update_yaxes(title_text=metric_column, row=row_idx, col=1)

    # --- Build dropdown menus ---
    act_buttons = [dict(
        label="All Activations",
        method="update",
        args=[{"visible": [True] * len(trace_activations)}],
    )]
    for act in activations:
        vis = [ta == act for ta in trace_activations]
        act_buttons.append(dict(
            label=act,
            method="update",
            args=[{"visible": vis}],
        ))

    lt_buttons = [dict(
        label="All Leadtimes",
        method="update",
        args=[{"visible": [True] * len(trace_leadtimes)}],
    )]
    for lt in leadtime_labels:
        vis = [tl == lt for tl in trace_leadtimes]
        lt_buttons.append(dict(
            label=lt,
            method="update",
            args=[{"visible": vis}],
        ))

    fig.update_layout(
        title=f"Hyperparameter Scatter: {metric_column}{title_suffix}",
        height=450 * len(x_axes),
        hovermode='closest',
        legend=dict(title="Activation | Leadtime", itemsizing='constant'),
        updatemenus=[
            dict(
                buttons=act_buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.08,
                yanchor="top",
                bgcolor="white",
                bordercolor="gray",
                font=dict(size=11),
            ),
            dict(
                buttons=lt_buttons,
                direction="down",
                showactive=True,
                x=0.25,
                xanchor="left",
                y=1.08,
                yanchor="top",
                bgcolor="white",
                bordercolor="gray",
                font=dict(size=11),
            ),
        ],
        annotations=[
            dict(text="Activation:", x=0.0, xref="paper", xanchor="left",
                 y=1.11, yref="paper", yanchor="bottom", showarrow=False,
                 font=dict(size=11)),
            dict(text="Leadtime:", x=0.25, xref="paper", xanchor="left",
                 y=1.11, yref="paper", yanchor="bottom", showarrow=False,
                 font=dict(size=11)),
        ],
    )

    return fig


def plot_scatter_v4(data: pd.DataFrame, metric_column: str, output_dir: Optional[str] = None) -> List[go.Figure]:
    """
    Generate scatter plots: one combined (all cycles) + one per cycle.
    Each has dropdown menus for filtering by activation and leadtime.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return []

    clean = data.dropna(subset=['leadtime']).copy()
    figures = []

    # Combined (all cycles)
    fig = _build_scatter_figure(clean, metric_column, title_suffix=" (All Cycles)")
    if output_dir:
        fig.write_html(f"{output_dir}/01_scatter_all_cycles.html")
        print("Saved: 01_scatter_all_cycles.html")
    figures.append(fig)

    # Per cycle
    cycles = sorted(clean['cycle'].unique())
    for cycle in cycles:
        cycle_data = clean[clean['cycle'] == cycle]
        c_int = int(cycle)
        fig = _build_scatter_figure(cycle_data, metric_column, title_suffix=f" (Cycle {c_int})")
        if output_dir:
            fig.write_html(f"{output_dir}/01_scatter_cycle_{c_int}.html")
            print(f"Saved: 01_scatter_cycle_{c_int}.html")
        figures.append(fig)

    return figures
