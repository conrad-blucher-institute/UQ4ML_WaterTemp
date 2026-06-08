"""
Wrapper: run inference + quick_viz for MSE/MAPE/CRPS using absolute-valued ESB data.

Behavior:
- Creates `data/ESB_datasets/esb_2020_2021_abs.csv` (if missing)
- Backs up original `esb_2020_2021.csv` to a timestamped .bak file
- Copies the abs file over the original path
- Runs inference for models: mse, mape, crps using existing runner
- Runs quick_viz for each model
- Restores original ESB CSV from backup (even on error)

Usage:
    python run_all_with_abs_data.py

Note: run from repo root (the script uses repo-relative paths).
"""
import subprocess
import shutil
from pathlib import Path
import time
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor

# --- CLI argument setup ---
parser = argparse.ArgumentParser(description="Run inference and visualization for multiple models with optional air-source suffix")
parser.add_argument("--air-source", type=str, default=None,
                    help="suffix for air temperature source (e.g., '_abs_air' -> results in mape_results_abs_air directory)")
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parent.parent
ESB_DIR = REPO_ROOT / "data" / "ESB_datasets"
SRC_ABS = ESB_DIR / "esb_2020_2021.csv"




MAKE_ABS_SCRIPT = ESB_DIR / "esb_2020_2021_make_new_dataset.py"

if not MAKE_ABS_SCRIPT.exists():
    print("Missing helper script: esb_2020_2021_make_new_dataset.py")
    sys.exit(1)

# Create abs CSV if needed
if not SRC_ABS.exists():
    print("Creating absolute-valued ESB CSV...")
    r = subprocess.run([sys.executable, str(MAKE_ABS_SCRIPT)], cwd=str(ESB_DIR))
    if r.returncode != 0:
        print("Failed to create abs CSV")
        sys.exit(1)
else:
    print(f"Found existing {SRC_ABS.name}")





def run_model(model_name):
    """Run inference and quick_viz for a single model."""
    print("\n" + "="*60)
    print(f"Running inference for model: {model_name} (using abs ESB file: {SRC_ABS})")
    infer_cmd = [
        sys.executable,
        "esb2020-2021_experiment/run_esb2020_2021_inference.py",
        "--model", model_name,
        "--esb-path", str(SRC_ABS)
    ]
    if args.air_source is not None:
        infer_cmd.extend(["--air-source", args.air_source])
    r = subprocess.run(infer_cmd, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        print(f"Inference script failed for model {model_name} (returncode={r.returncode})")
    else:
        print(f"Inference for {model_name} completed successfully")

    print(f"Running quick_viz for model: {model_name}")
    viz_cmd = [sys.executable, "esb2020-2021_experiment/quick_viz.py", "--model", model_name]
    if args.air_source is not None:
        viz_cmd.extend(["--air-source", args.air_source])
    r2 = subprocess.run(viz_cmd, cwd=str(REPO_ROOT))
    if r2.returncode != 0:
        print(f"quick_viz failed for model {model_name} (returncode={r2.returncode})")
    else:
        print(f"quick_viz for {model_name} completed successfully")

# Run models in parallel to speed up the process 
models = ["mse", "mape", "crps"]
with ThreadPoolExecutor(max_workers=len(models)) as executor:
    executor.map(run_model, models)

print("\nAll done.")
