"""
V6 Heatmap: grid layout — rows = parameter pairs, cols = leadtimes.
3 rows × 4 cols (swapped from v5). Cycle dropdown.
Local color scale. Auto-resize toggle.
"""

from typing import Optional
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .html_utils_v6 import wrap_figures_with_cycle_dropdown

HEATMAP_PAIRS = [
    ('activation', 'neurons', 'Activation', 'Hidden Units'),
    ('activation', 'num_layers', 'Activation', 'Num Layers'),
    ('num_layers', 'neurons', 'Num Layers', 'Hidden Units'),
]


def _build_heatmap_grid(data: pd.DataFrame, metric_column: str,
                         title_suffix: str = "") -> go.Figure:
    """Build grid: rows=param pairs, cols=leadtimes. Local color scale."""
    clean = data.dropna(subset=['leadtime']).copy()
    leadtimes = sorted(clean['leadtime'].unique())

    n_rows = len(HEATMAP_PAIRS)
    n_cols = len(leadtimes)

    subplot_titles = []
    for _, _, y_label, x_label in HEATMAP_PAIRS:
        for lt in leadtimes:
            subplot_titles.append(f"{y_label} vs {x_label} — LT {int(lt)}h")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    for row_idx, (y_col, x_col, y_label, x_label) in enumerate(HEATMAP_PAIRS, start=1):
        for col_idx, lt in enumerate(leadtimes, start=1):
            lt_data = clean[clean['leadtime'] == lt]

            pivot = lt_data.pivot_table(
                values=metric_column,
                index=y_col,
                columns=x_col,
                aggfunc='min',
            )

            try:
                pivot = pivot[sorted(pivot.columns, key=lambda x: float(x))]
            except (ValueError, TypeError):
                pass
            try:
                pivot = pivot.reindex(sorted(pivot.index, key=lambda x: float(x)))
            except (ValueError, TypeError):
                pass

            local_min = np.nanmin(pivot.values)
            local_max = np.nanmax(pivot.values)

            show_colorbar = (col_idx == n_cols and row_idx == 1)

            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=[str(c) for c in pivot.columns],
                    y=[str(i) for i in pivot.index],
                    colorscale='RdYlGn_r',
                    zmin=local_min,
                    zmax=local_max,
                    text=np.round(pivot.values, 4),
                    texttemplate='%{text:.3f}',
                    textfont={"size": 8},
                    showscale=show_colorbar,
                    colorbar=dict(title=metric_column) if show_colorbar else None,
                ),
                row=row_idx, col=col_idx,
            )

            fig.update_xaxes(
                title_text=x_label if row_idx == n_rows else "",
                row=row_idx, col=col_idx,
                type='category',
            )
            fig.update_yaxes(
                title_text=y_label if col_idx == 1 else "",
                row=row_idx, col=col_idx,
                type='category',
            )

    fig.update_layout(
        title=dict(
            text=(f"Heatmap Grid: Best {metric_column}{title_suffix}"
                  f"<br><sup style='color:#777;'>Each cell = best (min) {metric_column} "
                  f"for that parameter pair. Color scale is local per heatmap.</sup>"),
        ),
        height=280 * n_rows + 120,
        width=280 * n_cols + 120,
    )

    return fig


def plot_heatmap_v6(data: pd.DataFrame, metric_column: str,
                     output_dir: Optional[str] = None) -> Optional[str]:
    """
    Single HTML with cycle dropdown + auto-resize toggle.
    Grid: rows=param pairs, cols=leadtimes.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available.")
        return None

    clean = data.dropna(subset=['leadtime']).copy()
    cycles = sorted(clean['cycle'].unique())

    figure_htmls = {}

    fig = _build_heatmap_grid(clean, metric_column, " (All Cycles)")
    figure_htmls["All Cycles"] = fig.to_html(full_html=False, include_plotlyjs='cdn')

    for cycle in cycles:
        c_int = int(cycle)
        cycle_data = clean[clean['cycle'] == cycle]
        fig = _build_heatmap_grid(cycle_data, metric_column, f" (Cycle {c_int})")
        figure_htmls[f"Cycle {c_int}"] = fig.to_html(full_html=False, include_plotlyjs=False)

    base_html = wrap_figures_with_cycle_dropdown(figure_htmls, f"Heatmaps — {metric_column}")

    # Inject auto-resize toggle
    resize_inject = """
    <div class="filter-group">
        <label>Auto-resize:</label>
        <select id="resizeToggle" onchange="toggleResize()">
            <option value="fixed" selected>Fixed size</option>
            <option value="auto">Fit to window</option>
        </select>
    </div>
</div>
<script>
function toggleResize() {
    var mode = document.getElementById('resizeToggle').value;
    var plots = document.querySelectorAll('.plotly-graph-div');
    plots.forEach(function(gd) {
        if (mode === 'auto') {
            Plotly.relayout(gd, {width: null, height: null, autosize: true});
            gd.style.width = '100%';
        } else {
            // Re-render at original size
            window.dispatchEvent(new Event('resize'));
        }
    });
}
</script>"""

    # Insert before closing </div> of controls
    base_html = base_html.replace(
        """    </div>
</div>
""" + '<div id="cycle_view_0"',
        """    """ + resize_inject + '\n<div id="cycle_view_0"',
        1
    )

    if output_dir:
        path = f"{output_dir}/02_heatmaps.html"
        with open(path, 'w') as f:
            f.write(base_html)
        print("Saved: 02_heatmaps.html")

    return base_html
