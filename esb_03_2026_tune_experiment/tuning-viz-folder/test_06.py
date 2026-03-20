"""
Test the iteration comparison plot (06) with synthetic 2nd run.
Run 1 = real data, Run 2 = same data with val_mae + 13.
"""

from pathlib import Path
import pandas as pd
from tuning_viz.data_loader import TuningResultsLoader
from tuning_viz.plot_types.iteration_comparison_plot_v3 import plot_iteration_comparison_v3


def test_iteration_comparison(csv_path: str, metric_column: str, output_dir: str):
    loader = TuningResultsLoader(metric_column=metric_column)

    # Run 1: real data
    run1 = loader.load(csv_path, iteration_id=1)

    # Run 2: same data, metric + 13
    run2 = loader.load(csv_path, iteration_id=2)
    run2[metric_column] = run2[metric_column] + 13

    combined = pd.concat([run1, run2], ignore_index=True)

    print(f"Run 1 best {metric_column}: {run1[metric_column].min():.4f}")
    print(f"Run 2 best {metric_column} (fake +13): {run2[metric_column].min():.4f}")
    print(f"Iterations: {sorted(combined['iteration'].unique())}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_iteration_comparison_v3(combined, metric_column, output_dir=output_dir)


if __name__ == "__main__":
    test_iteration_comparison(
        csv_path='tune_experiment_results/mape_results_run1/mape_progress.csv',
        metric_column="val_mae",
        output_dir="./test_iteration_comparison",
    )
