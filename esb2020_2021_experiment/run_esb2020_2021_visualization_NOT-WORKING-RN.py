"""
ESB-2020-2021 Visualization Runner (Standalone)
================================================

Loads inference CSVs from experiment_results/crps_results/ and runs all three
visualization passes:
  - Pass 1: Creates per-cycle UQ summary files
  - Pass 2: Creates byCycle aggregate tables
  - Pass 3: Creates byUQMethod aggregate tables

This script handles ONLY visualization — inference must be run separately
with run_esb2020_2021_inference.py first.

Run from the repo root:
    python esb2020-2021_experiment/run_esb2020_2021_visualization.py

Input requirement:
    Inference CSVs must already exist under experiment_results/crps_results/

References:
    Viz logic       -> src/driver/visualization_driver.py
"""

"""
Hector Marrero-Colominas Comment, date: 2024-06-20

this file was trying replicate the visualization steps from visualization_driver.py, but it was not working correctly. There was too much hardcoded logic in the visualization functions that expected specific file paths and structures, which made it difficult to run them in a standalone script without the full experiment setup.
"""

import sys
from pathlib import Path

# --- Path setup ---
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))          # enables: from src.helper.utils_mse_crps import ...
sys.path.insert(0, str(_REPO_ROOT / "src"))  # enables: from evaluations.xxx import ...

from evaluations.cross_validation_visuals_paper import (
    mme_mse_crps_PNN_lead_times_singlePlot,
    decentralized_graphing_driver,
)
from evaluations.aggregate_tables import aggregateTable
from evaluations.boxplot_figures import (
    figure_5_plot,
    figure_6_7_plot,
    figure_13_plot,
    existance_checker,
)

# =============================================================================
# VISUALIZATION CONFIGURATION  — mirrors visualization_driver.py
# =============================================================================

import argparse

parser = argparse.ArgumentParser(description="Run visualization passes for a specified model type")
parser.add_argument("--model", choices=["crps", "mape", "mse"], default="crps",
                    help="model type (lowercase) used to locate inference files")
args = parser.parse_args()

model_name = args.model.upper()

# TODO: Verify these match the values used when training your models.
cycle_list      = [8]
start_iteration = 1
end_iteration   = 10           # inclusive
lead_time_list  = [12, 48, 96, 120]

save          = True
cycles        = cycle_list
leadTimes     = lead_time_list
architectures = [model_name.lower()]   # ["crps"]
iterations    = end_iteration
obsVsPred     = '2021'    # independent year label used by the visualization functions
expanded      = False
threshold     = 12
padding       = 24        # hours before/after Winter Storm URI to include

# =============================================================================
# VISUALIZATION PASS 1  — runAggregateCode = False
# Creates the per-cycle UQ files needed for the aggregate table functions.
# Mirrors:  visualization_driver.py  with  runAggregateCode = False
# =============================================================================

print("\n========== VIZ PASS 1: Plot Files (runAggregateCode=False) ==========\n")

# Use experiment_results/UQ_Files as the base directory for the visual-summary CSVs
# (independent of model type)
visualization_base = Path(__file__).resolve().parent / "experiment_results" / "UQ_Files"
visualization_base.mkdir(parents=True, exist_ok=True)

mme_mse_crps_PNN_lead_times_singlePlot(
    architectures,
    iterations,
    cycles,
    leadTimes,
    obsVsPred,
    expanded,
    base_dir=visualization_base,
)

for leadTime in leadTimes:
    decentralized_graphing_driver(architectures, leadTime, cycles, obsVsPred, save, input_path=visualization_base)

# =============================================================================
# VISUALIZATION PASS 2  — runAggregateCode = True, byCycle = True
# Creates byCycle aggregate tables (+ Figure 13 for the 2021 independent year).
# Mirrors:  visualization_driver.py  with  runAggregateCode = True, byCycle = True
# =============================================================================

print("\n========== VIZ PASS 2: byCycle Aggregate Tables (byCycle=True) ==========\n")

byCycle = True

if obsVsPred == '2021' and byCycle:
    aggregateTable(leadTimes, cycles, architectures, threshold, byCycle, obsVsPred, True, padding)
    figure_13_plot(padding)
elif obsVsPred == '2024' and byCycle:
    aggregateTable(leadTimes, cycles, architectures, threshold, byCycle, obsVsPred)
    figure_6_7_plot(obsVsPred, threshold)
elif obsVsPred == 'test' and byCycle:
    aggregateTable(leadTimes, cycles, architectures, threshold, byCycle, obsVsPred)
    figure_6_7_plot(obsVsPred, threshold)
else:
    aggregateTable(leadTimes, cycles, architectures, threshold, byCycle, obsVsPred)

# =============================================================================
# VISUALIZATION PASS 3  — runAggregateCode = True, byCycle = False
# Creates byUQMethod aggregate tables.
# Mirrors:  visualization_driver.py  with  runAggregateCode = True, byCycle = False
# =============================================================================

print("\n========== VIZ PASS 3: byUQMethod Aggregate Tables (byCycle=False) ==========\n")

byCycle = False

if obsVsPred == '2021' and not byCycle:
    aggregateTable(leadTimes, cycles, architectures, threshold, byCycle, obsVsPred, True, padding)
else:
    aggregateTable(leadTimes, cycles, architectures, threshold, byCycle, obsVsPred)

# =============================================================================
# OPTIONAL: Figure 5
# Requires aggregate tables for both 'val' and 'test' to already exist.
# Uncomment when those files are available.
# =============================================================================

# if existance_checker():
#     figure_5_plot()

print("\n\nVisualization Complete.")
