# Cleanup Plan for tuning-viz-folder

## Branch situation
- **Delete branch:** `esb_dev_add_training_trace` (its only unique commit `9802323` is superseded by the backup branch)
- **Work on:** `esb_dev_backup_2026320` (has all latest work including run_v16, training traces, etc.)

## Files to delete

### Old run scripts (tuning-viz-folder/)
Delete all except `run_v16.py`:
- run_v2.py, run_v3.py, run_v4.py, run_v5.py, run_v6.py, run_v7.py
- run_v8.py, run_v8_original.py, run_v9.py, run_v10.py, run_v11.py
- run_v12.py, run_v13.py, run_v14.py, run_v15.py

### Old plot modules (tuning_viz/plot_types/)

**Entirely unused modules (not imported by run_v16.py):**
- boxplot_plot.py, boxplot_plot_v2.py, boxplot_plot_v3.py, boxplot_plot_v5.py, boxplot_plot_v6.py, boxplot_plot_v7.py, boxplot_plot_v8.py
- scatter_plot.py, scatter_plot_v2.py, scatter_plot_v3.py, scatter_plot_v4.py, scatter_plot_v5.py, scatter_plot_v6.py, scatter_plot_v7.py, scatter_plot_v8.py
- layers_impact_plot.py, layers_impact_plot_v2.py, layers_impact_plot_v3.py, layers_impact_plot_v5.py
- violin_plot_v7.py, violin_plot_v8.py
- html_utils_v5.py, html_utils_v6.py, html_utils_v7.py, html_utils_v8.py

**Old versions of kept modules (run_v16.py imports the latest):**
- unified_plot: keep v15, delete v8, v9, v10, v11, v12, v13, v14
- heatmap_plot: keep v8, delete (none), v2, v3, v4, v5, v6, v7
- top_configs_plot: keep v12, delete (none), v2, v3, v5, v6, v7, v8
- parallel_coords: keep v8, delete v4, v5, v6, v7
- iteration_comparison_plot: keep v8, delete (none), v2, v3, v6, v7
- timeseries_compare: keep v16, delete v13, v14, v15

## Files to keep (tuning_viz/plot_types/)
- __init__.py
- unified_plot_v15.py
- heatmap_plot_v8.py
- top_configs_plot_v12.py
- parallel_coords_v8.py
- iteration_comparison_plot_v8.py
- timeseries_compare_v16.py

## Files to keep (tuning-viz-folder/)
- run_v16.py, run_all.py
- serve_timeseries.py, serve_timeseries_v2.py
- adapt_results.py, adapt_results_v2.py
- generate_predictions.py, benchmark_predict.py
- tuning_viz/data_loader.py

## Recovery
All deleted files can be recovered from git:
```bash
git checkout HEAD -- path/to/file.py
```
