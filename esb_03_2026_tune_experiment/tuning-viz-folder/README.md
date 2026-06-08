# Hyperparameter Tuning Visualization Suite

A modular, maintainable visualization system for analyzing ML hyperparameter tuning results across multiple iterations.

## Architecture

```
tuning_viz/
├── __init__.py                          # Package exports
├── data_loader.py                       # Data loading & parsing (80 lines)
├── viz_driver.py                        # Main orchestrator (70 lines)
├── plot_types/                          # Individual plot modules
│   ├── __init__.py                      # Plot function exports
│   ├── scatter_plot.py                  # Interactive scatter by leadtime
│   ├── heatmap_plot.py                  # Activation vs neurons heatmap
│   ├── boxplot_plot.py                  # Performance distribution
│   ├── top_configs_plot.py              # Top N configurations
│   ├── layers_impact_plot.py            # Layers effect analysis
│   └── iteration_comparison_plot.py     # Multi-run stability comparison
```

## Design Principles

### 1. **Modular by Plot Type**

Each visualization function lives in its own file. Adding a new plot type is as simple as creating a new file in `plot_types/`.

### 2. **Single Responsibility**

- `data_loader.py`: Only handles CSV loading & parsing
- Each `plot_types/*.py`: Only generates one specific visualization
- `viz_driver.py`: Only orchestrates and calls the other modules

### 3. **Easy Maintenance**

- Fix a bug in scatter plot? Open `scatter_plot.py` (not 600 lines)
- Update heatmap styling? Only touches that one file
- Add new metric? Minimal code changes needed

### 4. **Flexible Imports**

```python
# Use everything
from tuning_viz import viz_driver

# Use specific plots
from tuning_viz.plot_types import plot_scatter, plot_heatmap

# Use just the loader
from tuning_viz.data_loader import TuningResultsLoader
```

## Usage

### Simple: Run All Visualizations

```python
from tuning_viz import viz_driver

data, figs = viz_driver(
    csv_paths="mape_progress.csv",
    metric_column="val_mae",
    output_dir="./visualizations"
)
```

### Multiple Iterations (Compare Stability)

```python
data, figs = viz_driver(
    csv_paths=[
        "run1/mape_progress.csv",
        "run2/mape_progress.csv",
        "run3/mape_progress.csv"
    ],
    metric_column="val_mae",
    output_dir="./multi_iteration_results"
)
```

### Individual Plots Only

```python
from tuning_viz.plot_types import plot_scatter, plot_heatmap
from tuning_viz.data_loader import TuningResultsLoader

loader = TuningResultsLoader(metric_column="val_mae")
data = loader.load("mape_progress.csv")

# Generate specific plots
fig_scatter = plot_scatter(data, "val_mae", output_dir="./plots")
fig_heatmap = plot_heatmap(data, "val_mae", output_dir="./plots")
```

### Compare Multiple Optimization Types

```python
from tuning_viz import viz_driver

for opt_type in ['mape', 'mse', 'crps']:
    csv_path = f"results/{opt_type}_progress.csv"
    data, figs = viz_driver(
        csv_paths=csv_path,
        metric_column="val_mae",
        output_dir=f"./viz_{opt_type}"
    )
```

## Generated Visualizations

1. **Interactive Scatter** (`01_scatter_plot.html`)
   - Neurons vs metric, sized by layers, colored by activation
   - Faceted by leadtime
2. **Heatmaps** (`02_heatmap_leadtime_*.html`)
   - Activation functions vs neuron counts
   - One per leadtime
3. **Box Plot** (`03_boxplot_activation.html`)
   - Performance distribution by activation
   - Shows stability across configs
4. **Top Configurations** (`04_top_configurations.html`)
   - Bar chart of 15 best configs
   - Color-coded by performance
5. **Layers Impact** (`05_layers_impact.html`)
   - Effect of layer count on performance
   - Grouped by activation function
6. **Iteration Comparison** (`06_iteration_comparison.html`)
   - Best metric per iteration (if multi-run)
   - Shows consistency across runs

## Flexibility

### Different Metrics

```python
# Use val_mape instead of val_mae
viz_driver(csv_paths="...", metric_column="val_mape")

# Use val_loss
viz_driver(csv_paths="...", metric_column="val_loss")
```

### Output Format

```python
# Interactive plots (Plotly, default)
viz_driver(..., use_plotly=True)

# Static plots (PNG, matplotlib)
viz_driver(..., use_plotly=False)
```

### Custom Output Directory

```python
viz_driver(..., output_dir="./my_custom_results")
```

## Future Extensibility

### Add a New Plot Type

1. Create `tuning_viz/plot_types/my_new_plot.py`
2. Define function: `def plot_my_viz(data, metric_column, output_dir=None):`
3. Add to `plot_types/__init__.py`: `from .my_new_plot import plot_my_viz`
4. Call in `viz_driver.py` if needed

### Modify Existing Plot

- Open the specific file (e.g., `scatter_plot.py`)
- Make changes
- No need to touch other files

## Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `plotly` - Interactive visualizations (optional, falls back to matplotlib)
- `matplotlib` + `seaborn` - Static visualizations (optional, used if Plotly unavailable)

Install with:

```bash
pip install pandas numpy plotly matplotlib seaborn
```

## Why This Structure?

### Before (Monolithic)

- Single 600-line file
- Hard to find specific plot logic
- Risky to modify anything
- Testing individual plots = import everything
- Adding new plots feels overwhelming

### After (Modular)

- 6 focused plot files (~70 lines each)
- Find plot, open that file
- Safe to modify individual plots
- Can test each plot independently
- Adding new plots = add one new file
- Natural place for each concern
