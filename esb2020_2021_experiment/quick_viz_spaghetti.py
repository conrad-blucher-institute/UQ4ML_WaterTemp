"""
Quick Spaghetti Viz for ESB-2020-2021 inference CSVs
=====================================================

Plots each model's prediction as a separate line (spaghetti) so you can
inspect individual model behaviors. Also draws the ensemble mean and
an optional ±2σ band for context.

Usage:
    python quick_viz_spaghetti.py [--model crps|mape|mse] [--input-dir PATH] [--suffix name]

Outputs HTML files to `experiment_results/quick_viz_outputs_<model>/`.
"""

import argparse
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- CLI ---
parser = argparse.ArgumentParser(description="Quick spaghetti viz of ESB inference CSVs")
parser.add_argument("--model", choices=["crps", "mape", "mse"], default="crps")
parser.add_argument("--input-dir", type=Path, default=None,
                    help="override the default inference root (e.g. raw mape_results)")
parser.add_argument("--suffix", type=str, default=None,
                    help="optional output filename suffix (no leading underscore). Example: --suffix raw_orig")
args = parser.parse_args()

_REPO_ROOT = Path(__file__).resolve().parent.parent

if args.input_dir is not None:
    _RESULTS_DIR = args.input_dir
else:
    _RESULTS_DIR = Path(__file__).resolve().parent / "experiment_results" / f"{args.model}_results"

_OUTPUT_DIR = Path(__file__).resolve().parent / "experiment_results" / f"quick_viz_outputs_{args.model}"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# suffix handling
if args.suffix is not None:
    s = args.suffix
    if not s.startswith("_"):
        s = "_" + s
    _SUFFIX = s
else:
    _SUFFIX = ""

print(f"\nReading inference CSVs from: {_RESULTS_DIR}")
print(f"Saving plots to: {_OUTPUT_DIR}\n")

# --- Helpers ---

def load_predictions(results_dir):
    data_by_lead_time = {}
    csv_paths = sorted(glob.glob(str(results_dir / "*h" / "**" / "2021_datetime_obsv_predictions*.csv"), recursive=True))
    if not csv_paths:
        raise FileNotFoundError(f"No inference CSVs found in {results_dir}")
    for csv_path in csv_paths:
        lead_time_str = Path(csv_path).parts[-3]
        lead_time = int(lead_time_str.rstrip('h'))
        df = pd.read_csv(csv_path)
        df['date_time'] = pd.to_datetime(df['date_time'])
        data_by_lead_time.setdefault(lead_time, []).append((csv_path, df))
    print(f"Found {len(csv_paths)} inference CSVs across lead times: {sorted(data_by_lead_time.keys())}\n")
    return data_by_lead_time


def model_prediction_series(df):
    # Identify pred_* columns
    pred_cols = [c for c in df.columns if c.startswith('pred_')]
    if not pred_cols:
        raise ValueError("No pred_ columns found in dataframe")
    # If multiple pred columns (e.g. CRPS), average them to produce a single series
    if len(pred_cols) == 1:
        return df[pred_cols[0]].values
    else:
        return df[pred_cols].mean(axis=1).values


# --- Plotting ---

def plot_spaghetti(lead_time, model_series_list, dates, output_dir, suffix=''):
    fig = go.Figure()

    # Add each model line
    for idx, (label, series) in enumerate(model_series_list):
        fig.add_trace(go.Scatter(
            x=dates,
            y=series,
            mode='lines',
            name=f"Model {idx+1}: {label}",
            line=dict(width=1),
            opacity=0.6,
        ))

    # Ensemble mean
    all_preds = np.vstack([s for _, s in model_series_list])  # shape: (n_models, n_samples)
    ensemble_mean = np.mean(all_preds, axis=0)
    ensemble_std = np.std(all_preds, axis=0)

    fig.add_trace(go.Scatter(
        x=dates,
        y=ensemble_mean,
        mode='lines',
        name='Ensemble Mean',
        line=dict(color='black', width=3),
    ))

    # ±2σ band (transparent fill)
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates, dates[::-1]]),
        y=np.concatenate([ensemble_mean + 2*ensemble_std, (ensemble_mean - 2*ensemble_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(100,150,255,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='±2σ Band',
    ))

    fig.update_layout(
        title=f"Spaghetti Ensemble — Lead Time: {lead_time}h",
        xaxis_title='Date',
        yaxis_title='Temperature (°C)',
        hovermode='x unified',
        height=700,
    )

    out_path = output_dir / f"ensemble_spaghetti_{lead_time}h{suffix}.html"
    fig.write_html(out_path)
    print(f"  Saved: {out_path.name}")


# --- Main ---

data_by_lead = load_predictions(_RESULTS_DIR)

for lead_time in sorted(data_by_lead.keys()):
    print(f"Lead Time {lead_time}h:")
    entries = data_by_lead[lead_time]
    model_series_list = []
    for csv_path, df in entries:
        label = Path(csv_path).parent.name + '/' + Path(csv_path).name
        series = model_prediction_series(df)
        model_series_list.append((label, series))

    # Use dates from first model
    dates = entries[0][1]['date_time']

    plot_spaghetti(lead_time, model_series_list, dates, _OUTPUT_DIR, suffix=_SUFFIX)

print(f"\nAll spaghetti plots saved to: {_OUTPUT_DIR}")
