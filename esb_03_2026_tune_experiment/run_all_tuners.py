"""
Master Orchestrator for All Hyperparameter Tuners

Runs CRPS, MAPE, and MSE tuners sequentially with synchronized timing.
Aggregates results and provides comprehensive summary.
"""

"""
Usage examples (run from repo root):
python esb_03_2026_tune_experiment/run_all_tuners.py --workers 1 --repetitions 1

To start a temp monitor in another terminal (adjust threshold as needed), to throttle workers if overheating:
python esb_03_2026_tune_experiment/monitor/temp_monitor.py --output ./tmp_debug --interval 5 --threshold 85

For Debug:
python esb_03_2026_tune_experiment/run_all_tuners.py --debug --output ./tmp_debug --repetitions 3

For Debug with auto-workers:
python esb_03_2026_tune_experiment/run_all_tuners.py --auto-workers --output ./tmp_debug --repetitions 3
"""


import sys
import argparse
import csv
from pathlib import Path
from datetime import datetime
import time

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Suppress noisy TensorFlow logs and deprecation warnings at orchestrator level
import os
import warnings
import logging

# Reduce C++/native TF logging (0=all, 1=INFO, 2=WARNING, 3=ERROR)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
# Ignore Python DeprecationWarning messages (TF often emits these)
warnings.filterwarnings('ignore', category=DeprecationWarning)
# Quiet the tensorflow logger
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# If TensorFlow is importable, also set its internal logger to ERROR
try:
    import tensorflow as _tf
    try:
        _tf.get_logger().setLevel('ERROR')
    except Exception:
        pass
except Exception:
    # TF not installed in this environment; nothing to mute
    pass

# from esb_03_2026_tune_experiment.crps_tuner import CRPSTuner
from esb_03_2026_tune_experiment.mape_tuner import MAPETuner
from esb_03_2026_tune_experiment.mse_tuner import MSETuner
try:
    import psutil
except Exception:
    psutil = None


class TuningOrchestrator:
    """Orchestrates all three tuners with logging and progress tracking."""

    def __init__(self, output_dir: Path, max_workers: int = 4, verbose: int = 0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = max_workers
        self.verbose = verbose
        self.timing_log = self.output_dir / "orchestrator_timing.csv"
        self.summary_log = self.output_dir / "orchestrator_summary.txt"
        
        # Initialize CSV headers if file doesn't exist
        self._init_timing_csv()
    
    def _init_timing_csv(self):
        """Initialize timing CSV with headers."""
        if not self.timing_log.exists():
            with open(self.timing_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'event_type',
                    'tuner_name',
                    'duration_sec',
                    'notes'
                ])
    
    def _log_timing(self, event_type: str, tuner_name: str, duration_sec: float, notes: str = ""):
        """Log timing event."""
        with open(self.timing_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                event_type,
                tuner_name,
                f"{duration_sec:.2f}",
                notes
            ])
    
    def run_all_tuners(self, epoch_override: int = None, repetitions: int = 1, max_models: int = None):
        """Run all three tuners sequentially."""
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING ORCHESTRATOR")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print(f"Workers per tuner: {self.max_workers}")
        print(f"Start time: {datetime.now().isoformat()}\n")

        orchestrator_start = time.time()
        results = {}

        # Loop over repetitions: run 1 for each tuner, then run 2, etc.
        for run_idx in range(int(repetitions)):
            run_id = run_idx + 1

            # MAPE
            print(f"Running MAPE tuner (run {run_id})...")
            mape_output = self.output_dir / f"mape_results_run{run_id}"
            try:
                start = time.time()



                # Respect external throttle flag if present (monitor may toggle)
                cur_workers = self.max_workers
                throttle_flag = self.output_dir / 'throttle.flag'
                if throttle_flag.exists():
                    cur_workers = max(1, cur_workers // 2)
                    print(f"Throttle flag present, reducing workers to {cur_workers}")

                mape_tuner = MAPETuner(mape_output, max_workers=cur_workers, keras_save_dir=mape_output / "keras_files", verbose=self.verbose)


                if epoch_override is not None:
                    mape_tuner.default_epochs = int(epoch_override)
                mape_tuner.run_tuning(run_num=run_id, max_configs=max_models)
                duration = time.time() - start
                self._log_timing('tuner_complete', f'MAPE_run{run_id}', duration, 'completed successfully')
                results.setdefault('MAPE', []).append({'run': run_id, 'status': 'success', 'duration': duration})
                print(f"✓ MAPE run {run_id} completed in {duration:.1f}s\n")
            except Exception as e:
                duration = time.time() - start
                self._log_timing('tuner_error', f'MAPE_run{run_id}', duration, str(e))
                results.setdefault('MAPE', []).append({'run': run_id, 'status': 'error', 'duration': duration, 'error': str(e)})
                print(f"✗ MAPE run {run_id} failed: {e}\n")

            # MSE
            print(f"Running MSE tuner (run {run_id})...")
            mse_output = self.output_dir / f"mse_results_run{run_id}"
            try:
                start = time.time()



                # Respect throttle flag here too
                cur_workers = self.max_workers
                throttle_flag = self.output_dir / 'throttle.flag'
                if throttle_flag.exists():
                    cur_workers = max(1, cur_workers // 2)
                    print(f"Throttle flag present, reducing workers to {cur_workers}")

                mse_tuner = MSETuner(mse_output, max_workers=cur_workers, keras_save_dir=mse_output / "keras_files", verbose=self.verbose)


                if epoch_override is not None:
                    mse_tuner.default_epochs = int(epoch_override)
                mse_tuner.run_tuning(run_num=run_id, max_configs=max_models)
                duration = time.time() - start
                self._log_timing('tuner_complete', f'MSE_run{run_id}', duration, 'completed successfully')
                results.setdefault('MSE', []).append({'run': run_id, 'status': 'success', 'duration': duration})
                print(f"✓ MSE run {run_id} completed in {duration:.1f}s\n")
            except Exception as e:
                duration = time.time() - start
                self._log_timing('tuner_error', f'MSE_run{run_id}', duration, str(e))
                results.setdefault('MSE', []).append({'run': run_id, 'status': 'error', 'duration': duration, 'error': str(e)})
                print(f"✗ MSE run {run_id} failed: {e}\n")

        total_duration = time.time() - orchestrator_start
        self._log_timing('orchestrator_complete', 'ALL', total_duration, 'all tuners finished')

        # Print summary
        self._print_summary(results, total_duration)
    
    def _print_summary(self, results: dict, total_duration: float):
        """Print and save summary report."""
        summary = []
        summary.append("=" * 80)
        summary.append("TUNING ORCHESTRATOR SUMMARY")
        summary.append("=" * 80)
        summary.append(f"End time: {datetime.now().isoformat()}")
        summary.append(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        summary.append("")
        
        for model in ['CRPS', 'MAPE', 'MSE']:
            if model in results:
                # results[model] is a list of run dicts
                for r in results[model]:
                    status = str(r.get('status', 'unknown')).upper()
                    duration = float(r.get('duration', 0.0))
                    run = int(r.get('run', 0))
                    if status == 'SUCCESS':
                        summary.append(f"SUCCESS {model} run {run}: {status:8s} ({duration:7.1f}s)")
                    else:
                        summary.append(f"FAILURE {model} run {run}: {status:8s} ({duration:7.1f}s)")
                        if 'error' in r:
                            summary.append(f"  Error: {r['error']}")
        
        summary.append("")
        summary.append(f"Output directory: {self.output_dir}")
        summary.append(f"Timing log: {self.timing_log}")
        summary.append(f"Summary log: {self.summary_log}")
        summary.append("")
        summary.append("For detailed timing analysis, see orchestrator_timing.csv")
        summary.append("For model-specific progress, check individual result directories:")
        summary.append(f"  - {self.output_dir / 'crps_results'}")
        summary.append(f"  - {self.output_dir / 'mape_results'}")
        summary.append(f"  - {self.output_dir / 'mse_results'}")
        summary.append("=" * 80)
        
        summary_text = "\n".join(summary)
        print("\n" + summary_text)
        
        # Save summary
        with open(self.summary_log, 'w') as f:
            f.write(summary_text + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Master Orchestrator for Hyperparameter Tuning"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: results/esb_tuner for production, results/esb_tuner_debug_MM-DD-YYYY_HHhMMmSSs for --debug)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers per tuner (default: 1 for --debug, 4 for production)"
    )
    parser.add_argument(
        "--auto-workers",
        action='store_true',
        help="Auto-select workers based on physical cores (recommended for CPU-bound workloads)"
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="Run in debug mode (2 epochs, 10 models, 1 worker, 2 reps — all overridable)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override default epochs for tuners (useful with --debug)"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Number of repeated runs (default: 2 for --debug, 1 for production)"
    )
    parser.add_argument(
        "--max_models",
        type=int,
        default=None,
        help="Maximum number of models to run per tuner (default: 10 for --debug, all for production)"
    )
    args = parser.parse_args()

    # --- Apply debug defaults (CLI flags always take priority) ---
    if args.debug:
        epoch_override = args.epochs if args.epochs is not None else 2
        max_models = args.max_models if args.max_models is not None else 10
        workers = args.workers if args.workers is not None else 1
        repetitions = args.repetitions if args.repetitions is not None else 2
        if args.output is None:
            timestamp = datetime.now().strftime('%m-%d-%Y_%Hh%Mm%Ss')
            args.output = _REPO_ROOT / "results" / f"esb_tuner_debug_{timestamp}"
    else:
        epoch_override = args.epochs  # None unless explicitly set
        max_models = args.max_models  # None = run all
        workers = args.workers if args.workers is not None else 4
        repetitions = args.repetitions if args.repetitions is not None else 1
        if args.output is None:
            args.output = _REPO_ROOT / "results" / "esb_tuner"
    if args.auto_workers:
        try:
            if psutil is not None:
                physical = psutil.cpu_count(logical=False) or 1
            else:
                physical = os.cpu_count() or 1
        except Exception:
            physical = os.cpu_count() or 1
        workers = max(1, int(physical) - 1)
        print(f"Auto-workers enabled: detected {physical} physical cores -> using {workers} workers")

    # Ensure BLAS/OpenMP thread limits to avoid oversubscription
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

    verbose = 2 if args.debug else 0
    # Set env var so preparingData() debug logs land in the experiment folder
    os.environ['TUNER_DEBUG_LOG_DIR'] = str(args.output)
    orchestrator = TuningOrchestrator(args.output, max_workers=workers, verbose=verbose)
    orchestrator.run_all_tuners(epoch_override=epoch_override, repetitions=repetitions, max_models=max_models)
