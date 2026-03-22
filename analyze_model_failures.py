#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Failure Analysis Script
Analyzes model predictions for extreme low temperatures and other metrics.
Identifies models that predict unrealistic water temperatures (e.g., < -1\u00b0C).

@author: Generated for UQ4ML_WaterTemp
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from evaluations.cross_validation_visuals_paper import model_parser, model_selection_conditional

######## Configuration ########

# Cycles to process
cycles = [0, 1, 2, 3]

# Lead times to process
leadTimes = [12, 48, 96, 120]

# Architecture
architectures = ["mse"]

# Number of iterations
iterations = 30

# Data type
obsVsPred = '2021'

# Temperature threshold for "extreme low" failure
extreme_low_threshold = -5.0

# Threshold for "concerning" predictions (but not extreme)
concerning_threshold = 5.0

# Output directory for reports
output_directory = "model_failure_analysis"


def analyze_predictions(df_combined, leadTime, cycle, architecture):
    """
    Analyze model predictions for extreme values and other issues.
    
    Parameters:
    -----------
    df_combined : pd.DataFrame
        Combined predictions from all iterations
    leadTime : int
        Lead time in hours
    cycle : int
        Cycle number
    architecture : str
        Model architecture
    
    Returns:
    --------
    dict with analysis results
    """
    
    # Get target (observed water temperature)
    observed = df_combined['target']
    
    # Get all prediction columns
    pred_columns = sorted(
        [col for col in df_combined.columns if '_iteration_' in col],
        key=lambda x: int(x.split('_iteration_')[-1])
    )
    
    if not pred_columns:
        return None
    
    # Extract predictions
    predictions = df_combined[pred_columns]
    
    # Find extreme lows per iteration
    extreme_low_per_iteration = []
    min_per_iteration = []
    mean_per_iteration = []
    max_per_iteration = []
    concerning_low_per_iteration = []
    
    for col in pred_columns:
        pred_values = df_combined[col]
        
        # Count how many predictions are below extreme threshold
        extreme_count = (pred_values < extreme_low_threshold).sum()
        extreme_low_per_iteration.append(extreme_count)
        
        # Count concerning lows (not extreme, but still low)
        concerning_count = ((pred_values < concerning_threshold) & (pred_values >= extreme_low_threshold)).sum()
        concerning_low_per_iteration.append(concerning_count)
        
        # Statistics
        min_per_iteration.append(pred_values.min())
        mean_per_iteration.append(pred_values.mean())
        max_per_iteration.append(pred_values.max())
    
    # Count how many iterations had ANY extreme low predictions
    models_with_extreme = sum(1 for count in extreme_low_per_iteration if count > 0)
    models_with_concerning = sum(1 for count in concerning_low_per_iteration if count > 0)
    
    # Total extreme low predictions across all models and time steps
    total_extreme_predictions = sum(extreme_low_per_iteration)
    total_timepoints = len(observed) * iterations
    
    # Observed min/max for reference
    obs_min = observed.min()
    obs_max = observed.max()
    obs_mean = observed.mean()
    
    return {
        'leadTime': leadTime,
        'cycle': cycle,
        'architecture': architecture,
        'iterations': iterations,
        'timesteps': len(observed),
        
        # Extreme low analysis
        'models_with_extreme': models_with_extreme,
        'models_with_extreme_pct': (models_with_extreme / iterations) * 100,
        'models_with_concerning': models_with_concerning,
        'models_with_concerning_pct': (models_with_concerning / iterations) * 100,
        
        # Total predictions
        'total_extreme_predictions': total_extreme_predictions,
        'total_predictions': total_timepoints,
        'extreme_predictions_pct': (total_extreme_predictions / total_timepoints) * 100,
        
        # Per-iteration statistics
        'mean_extreme_count': np.mean(extreme_low_per_iteration),
        'mean_concerning_count': np.mean(concerning_low_per_iteration),
        'min_per_iteration': min_per_iteration,
        'mean_per_iteration': mean_per_iteration,
        'max_per_iteration': max_per_iteration,
        
        # Ensemble statistics
        'ensemble_min': predictions.min().min(),
        'ensemble_mean': predictions.mean().mean(),
        'ensemble_max': predictions.max().max(),
        
        # Observed statistics
        'observed_min': obs_min,
        'observed_mean': obs_mean,
        'observed_max': obs_max,
    }


def print_analysis_report(analysis_results):
    """
    Print a formatted analysis report for the given results.
    """
    
    print("\n" + "="*80)
    print(f"ANALYSIS: {analysis_results['architecture'].upper()} - "
          f"LeadTime {analysis_results['leadTime']}h - Cycle {analysis_results['cycle']}")
    print("="*80)
    
    # Extreme low failures
    print(f"\n EXTREME LOW FAILURES (< {extreme_low_threshold} C):")
    print(f"  Models affected: {analysis_results['models_with_extreme']}/{analysis_results['iterations']} "
          f"({analysis_results['models_with_extreme_pct']:.1f}%)")
    print(f"  Total extreme predictions: {analysis_results['total_extreme_predictions']:,} / "
          f"{analysis_results['total_predictions']:,} "
          f"({analysis_results['extreme_predictions_pct']:.2f}%)")
    print(f"  Avg extreme predictions per affected model: {analysis_results['mean_extreme_count']:.2f}")
    
    # Concerning lows
    print(f"\n  CONCERNING LOWS ({extreme_low_threshold}\u00b0C to {concerning_threshold}\u00b0C):")
    print(f"  Models affected: {analysis_results['models_with_concerning']}/{analysis_results['iterations']} "
          f"({analysis_results['models_with_concerning_pct']:.1f}%)")
    print(f"  Avg concerning predictions per affected model: {analysis_results['mean_concerning_count']:.2f}")
    
    # Per-iteration range
    print(f"\n PER-MODEL STATISTICS:")
    best_model_idx = np.argmin([min(mins) for mins in [analysis_results['min_per_iteration']]])
    worst_model_idx = np.argmin([max_ for max_ in analysis_results['max_per_iteration']])
    
    print(f"  Iteration with best min: #{best_model_idx + 1} "
          f"(min: {analysis_results['min_per_iteration'][best_model_idx]:.2f}\u00b0C)")
    print(f"  Iteration with worst min: #{worst_model_idx + 1} "
          f"(min: {analysis_results['min_per_iteration'][worst_model_idx]:.2f}\u00b0C)")
    
    min_acrosss_all_iterations = min(analysis_results['min_per_iteration'])
    max_across_all_iterations = max(analysis_results['max_per_iteration'])
    mean_across_all_iterations = np.mean(analysis_results['mean_per_iteration'])
    
    print(f"  Min across all models: {min_acrosss_all_iterations:.2f}\u00b0C")
    print(f"  Max across all models: {max_across_all_iterations:.2f}\u00b0C")
    print(f"  Mean across all models: {mean_across_all_iterations:.2f}\u00b0C")
    
    # Ensemble statistics
    print(f"\n ENSEMBLE STATISTICS:")
    print(f"  Ensemble min: {analysis_results['ensemble_min']:.2f}\u00b0C")
    print(f"  Ensemble mean: {analysis_results['ensemble_mean']:.2f}\u00b0C")
    print(f"  Ensemble max: {analysis_results['ensemble_max']:.2f}\u00b0C")
    
    # Observed reference
    print(f"\n OBSERVED DATA (for reference):")
    print(f"  Observed min: {analysis_results['observed_min']:.2f}\u00b0C")
    print(f"  Observed mean: {analysis_results['observed_mean']:.2f}\u00b0C")
    print(f"  Observed max: {analysis_results['observed_max']:.2f}\u00b0C")
    
    # Performance classification
    print(f"\n PERFORMANCE CLASSIFICATION:")
    if analysis_results['models_with_extreme_pct'] == 0:
        print(f"   EXCELLENT: No models predicted extreme lows")
    elif analysis_results['models_with_extreme_pct'] < 10:
        print(f"   GOOD: Only {analysis_results['models_with_extreme']}/{analysis_results['iterations']} models with extreme lows")
    elif analysis_results['models_with_extreme_pct'] < 50:
        print(f"    CONCERNING: {analysis_results['models_with_extreme']}/{analysis_results['iterations']} models with extreme lows")
    else:
        print(f"   POOR: Majority ({analysis_results['models_with_extreme']}/{analysis_results['iterations']}) have extreme lows")


def main():
    """
    Main function to run the analysis.
    """
    
    # Create output directory if it doesn't exist
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("MODEL FAILURE ANALYSIS - EXTREME LOW TEMPERATURE DETECTION")
    print("="*80)
    print(f"Analyzing for temperatures below {extreme_low_threshold}\u00b0C")
    print(f"Concerning threshold: {concerning_threshold}\u00b0C")
    print(f"Output directory: {Path(output_directory).resolve()}")
    
    all_results = []
    detailed_reports = []
    
    # Loop through all configurations
    for architecture in architectures:
        for cycle in cycles:
            for leadTime in leadTimes:
                
                try:
                    # Get model names
                    model_names = model_selection_conditional(leadTime, architecture)
                    
                    if not model_names:
                        print(f"\n  No model found for {architecture} at {leadTime}h")
                        continue
                    
                    model = model_names[0]
                    
                    # Determine directory
                    main_dir = f"results/{architecture.lower()}_results"
                    
                    # Load data
                    df_combined = model_parser(
                        MAIN_DIRECTORY=main_dir,
                        model=model,
                        architecture=architecture,
                        obsVsPred=obsVsPred,
                        iterations=iterations,
                        cycle=cycle,
                        leadTime=leadTime
                    )
                    
                    # Analyze
                    analysis = analyze_predictions(df_combined, leadTime, cycle, architecture)
                    
                    if analysis:
                        all_results.append(analysis)
                        report = _generate_detailed_report(analysis)
                        print(report)
                        detailed_reports.append(report)
                    
                except Exception as e:
                    print(f"\n Error processing {architecture} - {leadTime}h - Cycle {cycle}: {str(e)}")
    
    # Save detailed reports to text file
    detailed_report_path = Path(output_directory) / f"detailed_analysis_{obsVsPred}_{architecture}.txt"
    with open(detailed_report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(detailed_reports))
    
    print(f"\n Detailed reports saved to: {detailed_report_path.resolve()}")
    
    # Summary table
    print("\n\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    summary_df = pd.DataFrame([
        {
            'Architecture': r['architecture'].upper(),
            'LeadTime': f"{r['leadTime']}h",
            'Cycle': r['cycle'],
            'Extreme Models': f"{r['models_with_extreme']}/{r['iterations']} ({r['models_with_extreme_pct']:.0f}%)",
            'Total Extreme Preds': f"{r['total_extreme_predictions']:,}",
            'Ensemble Min': f"{r['ensemble_min']:.2f}\u00b0C",
            'Observed Min': f"{r['observed_min']:.2f}\u00b0C",
        }
        for r in all_results
    ])
    
    print(summary_df.to_string(index=False))
    
    # Save summary table to CSV
    csv_path = Path(output_directory) / f"summary_table_{obsVsPred}_{architecture}.csv"
    summary_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\ Summary table saved to CSV: {csv_path.resolve()}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


def _generate_detailed_report(analysis_results):
    """
    Generate a detailed report string for the given analysis results.
    """
    
    report_lines = []
    
    report_lines.append("\n" + "="*80)
    report_lines.append(f"ANALYSIS: {analysis_results['architecture'].upper()} - "
          f"LeadTime {analysis_results['leadTime']}h - Cycle {analysis_results['cycle']}")
    report_lines.append("="*80)
    
    # Extreme low failures
    report_lines.append(f"\n EXTREME LOW FAILURES (< {extreme_low_threshold}\u00b0C):")
    report_lines.append(f"  Models affected: {analysis_results['models_with_extreme']}/{analysis_results['iterations']} "
          f"({analysis_results['models_with_extreme_pct']:.1f}%)")
    report_lines.append(f"  Total extreme predictions: {analysis_results['total_extreme_predictions']:,} / "
          f"{analysis_results['total_predictions']:,} "
          f"({analysis_results['extreme_predictions_pct']:.2f}%)")
    report_lines.append(f"  Avg extreme predictions per affected model: {analysis_results['mean_extreme_count']:.2f}")
    
    # Concerning lows
    report_lines.append(f"\n  CONCERNING LOWS ({extreme_low_threshold}\u00b0C to {concerning_threshold}\u00b0C):")
    report_lines.append(f"  Models affected: {analysis_results['models_with_concerning']}/{analysis_results['iterations']} "
          f"({analysis_results['models_with_concerning_pct']:.1f}%)")
    report_lines.append(f"  Avg concerning predictions per affected model: {analysis_results['mean_concerning_count']:.2f}")
    
    # Per-iteration range
    report_lines.append(f"\n PER-MODEL STATISTICS:")
    best_model_idx = np.argmin([min(mins) for mins in [analysis_results['min_per_iteration']]])
    worst_model_idx = np.argmin([max_ for max_ in analysis_results['max_per_iteration']])
    
    report_lines.append(f"  Iteration with best min: #{best_model_idx + 1} "
          f"(min: {analysis_results['min_per_iteration'][best_model_idx]:.2f}\u00b0C)")
    report_lines.append(f"  Iteration with worst min: #{worst_model_idx + 1} "
          f"(min: {analysis_results['min_per_iteration'][worst_model_idx]:.2f}\u00b0C)")
    
    min_acrosss_all_iterations = min(analysis_results['min_per_iteration'])
    max_across_all_iterations = max(analysis_results['max_per_iteration'])
    mean_across_all_iterations = np.mean(analysis_results['mean_per_iteration'])
    
    report_lines.append(f"  Min across all models: {min_acrosss_all_iterations:.2f}\u00b0C")
    report_lines.append(f"  Max across all models: {max_across_all_iterations:.2f}\u00b0C")
    report_lines.append(f"  Mean across all models: {mean_across_all_iterations:.2f}\u00b0C")
    
    # Ensemble statistics
    report_lines.append(f"\n ENSEMBLE STATISTICS:")
    report_lines.append(f"  Ensemble min: {analysis_results['ensemble_min']:.2f}\u00b0C")
    report_lines.append(f"  Ensemble mean: {analysis_results['ensemble_mean']:.2f}\u00b0C")
    report_lines.append(f"  Ensemble max: {analysis_results['ensemble_max']:.2f}\u00b0C")
    
    # Observed reference
    report_lines.append(f"\n OBSERVED DATA (for reference):")
    report_lines.append(f"  Observed min: {analysis_results['observed_min']:.2f}\u00b0C")
    report_lines.append(f"  Observed mean: {analysis_results['observed_mean']:.2f}\u00b0C")
    report_lines.append(f"  Observed max: {analysis_results['observed_max']:.2f}\u00b0C")
    
    # Performance classification
    report_lines.append(f"\n PERFORMANCE CLASSIFICATION:")
    if analysis_results['models_with_extreme_pct'] == 0:
        report_lines.append(f"   EXCELLENT: No models predicted extreme lows")
    elif analysis_results['models_with_extreme_pct'] < 10:
        report_lines.append(f"  ✓ GOOD: Only {analysis_results['models_with_extreme']}/{analysis_results['iterations']} models with extreme lows")
    elif analysis_results['models_with_extreme_pct'] < 50:
        report_lines.append(f"    CONCERNING: {analysis_results['models_with_extreme']}/{analysis_results['iterations']} models with extreme lows")
    else:
        report_lines.append(f"   POOR: Majority ({analysis_results['models_with_extreme']}/{analysis_results['iterations']}) have extreme lows")
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    main()
