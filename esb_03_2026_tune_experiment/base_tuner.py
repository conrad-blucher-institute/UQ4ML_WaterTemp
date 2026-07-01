"""
Base hyperparameter tuner template with grid search, multiprocessing, and progress tracking.
Extend this class for specific model types (CRPS, MAPE, MSE).
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tuner_utils import (
    GridSearchConfig, ProgressTracker, TimingLogger
)

class BaseHyperparameterTuner:
    """
    Base class for hyperparameter tuning with grid search, multiprocessing, and tracking.
    
    Extend this class and implement:
    - `_build_model(config)`: create a model from configuration
    - `_train_model(model, config)`: train the model and return loss
    - `get_grid_search_config()`: return GridSearchConfig instance
    """
    
    def __init__(self, model_type: str, output_dir: Path, max_workers: int = 4, run_num: int = 0, metrics: List[str] = None, default_epochs: int = 200000, keras_save_dir: Path = None, verbose: int = 0, grid_config: 'GridSearchConfig' = None, config_overrides: Dict[str, Any] = None):
        """
        Args:
            model_type: 'CRPS', 'MAPE', or 'MSE'
            output_dir: directory to save progress and logs
            max_workers: number of parallel workers for multiprocessing
            grid_config: optional GridSearchConfig that OVERRIDES the subclass's
                hardcoded get_grid_search_config(). Used by the esb entry layer to
                drive the grid from the Config single-source-of-truth. When None
                (the run_all_tuners.py path), the subclass's grid is used unchanged.
            config_overrides: optional dict of extra keys injected into every
                per-config dict (e.g. path_to_data, input_structure, callback
                patiences). Injected with setdefault so explicit per-config keys
                win; absent → unchanged legacy behavior.
        """
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.run_num = int(run_num)
        # default epochs to inject into configs when not present
        self.default_epochs = int(default_epochs)
        # Metrics that will be recorded for each tuning run (strings or callables handled by subclass)
        self.metrics = metrics or []
        self.verbose = int(verbose)
        # Optional external grid + per-config overrides (esb entry layer).
        self.grid_config = grid_config
        self.config_overrides = dict(config_overrides) if config_overrides else None

        # Directory to save trained .keras files (None = don't save)
        self.keras_save_dir = Path(keras_save_dir) if keras_save_dir is not None else None
        if self.keras_save_dir is not None:
            self.keras_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.progress_path = self.output_dir / f"{model_type.lower()}_progress.csv"
        self.timing_log_path = self.output_dir / f"{model_type.lower()}_timing.csv"
        
        self.progress_tracker = ProgressTracker(self.progress_path)
        self.timing_logger = TimingLogger(self.timing_log_path)
    
    def get_grid_search_config(self) -> GridSearchConfig:
        """Override this to define grid search parameters for your model."""
        raise NotImplementedError("Subclasses must implement get_grid_search_config()")
    
    def _build_model(self, config: Dict[str, Any]):
        """Override this to build a model from configuration."""
        raise NotImplementedError("Subclasses must implement _build_model()")

    def _prepare_data(self, config: Dict[str, Any]):
        """Hook to prepare data for a specific configuration.

        Default implementation is a no-op. Subclasses can override this to call
        the project's `preparingData()` (or equivalent) and return whatever that
        function provides (e.g., (X_train, y_train, X_val, y_val)).
        """
        return None
    
    def _train_model(self, model, config: Dict[str, Any]) -> float:
        """Override this to train the model and return loss value."""
        raise NotImplementedError("Subclasses must implement _train_model()")
    
    def _tune_single_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tune a single configuration. Called by multiprocessing workers.
        """
        try:
            print(f"  TUNING: {config}")
            
            # Defer model construction to the tuner's _train_model implementation.
            # Subclasses should build the model there (they may need data to
            # infer input_shape) and may ignore the `model` argument.
            model = None
            build_time = 0.0
            
            # Time the training
            train_start = time.time()
            train_result = self._train_model(model, config)
            # _train_model may return a float loss, or (loss, metrics_dict)
            if isinstance(train_result, tuple) and len(train_result) == 2 and isinstance(train_result[1], dict):
                loss_value, metrics_dict = train_result
            else:
                loss_value = train_result
                metrics_dict = None
            train_time = time.time() - train_start
            self.timing_logger.log_component(
                f"train_{config['activation']}_{config['num_layers']}_{config['neurons']}",
                train_time
            )
            
            # Log result
            self.progress_tracker.log_result(
                config['model_type'], config['lead_time'], config['cycle'],
                config['activation'], config['num_layers'], config['neurons'],
                loss_value, status="completed", run_num=self.run_num, metrics=metrics_dict,
                dropout=config.get('dropout', 0.0)
            )
            
            config['loss_value'] = loss_value
            config['build_time'] = build_time
            config['train_time'] = train_time
            config['status'] = 'completed'
            
            return config
            
        except Exception as e:
            print(f"  ERROR: {config} - {str(e)}")
            print(traceback.format_exc())
            
            self.progress_tracker.log_result(
                config['model_type'], config['lead_time'], config['cycle'],
                config['activation'], config['num_layers'], config['neurons'],
                loss_value=-1.0, status=f"error: {str(e)}", run_num=self.run_num,
                dropout=config.get('dropout', 0.0)
            )
            config['status'] = 'error'
            config['error'] = str(e)
            return config
    
    def run_tuning(self, run_num: int = 0, max_configs: int = None):
        """Run the full hyperparameter tuning grid search.

        Args:
            run_num: integer run identifier; injected into each config and used
                by `ProgressTracker` to allow multiple independent runs.
            max_configs: if set, limit the number of configs explored (useful
                for fast debug runs).
        """
        print(f"\n{'='*80}")
        print(f"Starting hyperparameter tuning for {self.model_type}")
        print(f"{'='*80}\n")
        
        overall_start = time.time()
        # set run number for this tuning session
        self.run_num = int(run_num)
        
        # Get grid search configuration. An externally supplied grid (from the
        # esb entry layer) takes precedence over the subclass's hardcoded grid.
        grid_config = self.grid_config if self.grid_config is not None else self.get_grid_search_config()
        configs = grid_config.generate_configs()

        # Inject externally supplied per-config overrides FIRST (so the base
        # defaults below only fill what the override didn't set). setdefault keeps
        # generate_configs' own keys (run_num, lead_time, ...) authoritative.
        if self.config_overrides:
            for c in configs:
                for k, v in self.config_overrides.items():
                    c.setdefault(k, v)

        # Inject default epochs into configs if not explicitly set
        if hasattr(self, 'default_epochs') and self.default_epochs is not None:
            for c in configs:
                if 'epochs' not in c:
                    c['epochs'] = int(self.default_epochs)

        # Ensure each config records the active run number
        for c in configs:
            c['run_num'] = int(self.run_num)

        # Inject keras save dir so workers know where to save models
        for c in configs:
            c['keras_save_dir'] = str(self.keras_save_dir) if self.keras_save_dir is not None else None

        # Inject common defaults required by _build_model implementations
        for c in configs:
            if 'output_units' not in c:
                c['output_units'] = 1
            if 'output_activation' not in c:
                c['output_activation'] = 'linear'
            if 'learning_rate' not in c:
                c['learning_rate'] = 0.001
            if 'verbose' not in c:
                c['verbose'] = self.verbose

        total_before = len(configs)
        # Optionally limit to a subset (debug)
        if max_configs is not None:
            configs = configs[:int(max_configs)]

        # Filter out already-completed configs BEFORE submitting to workers
        already_done = len(self.progress_tracker.completed)
        configs = [
            c for c in configs
            if not self.progress_tracker.is_completed(
                c['model_type'], c['lead_time'], c['cycle'],
                c['activation'], c['num_layers'], c['neurons'],
                dropout=c.get('dropout', 0.0), run_num=c.get('run_num', 0)
            )
        ]
        total_configs = len(configs)

        if max_configs is not None:
            print(f"Total configurations (limited from {total_before})")
        print(f"Already completed: {already_done}")
        print(f"Remaining to run: {total_configs}")
        if total_configs == 0:
            print("Nothing to do — all configs already completed.\n")
            return
        print(f"Using {self.max_workers} parallel workers\n")
        
        completed_count = 0
        
        # Submit futures in batches to avoid holding all configs in memory at once.
        # Each batch is at most (max_workers * 2) configs so the executor always
        # has work queued without bloating memory.
        batch_size = max(1, self.max_workers * 2)
        for batch_start in range(0, total_configs, batch_size):
            batch = configs[batch_start : batch_start + batch_size]

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._tune_single_config, config): config for config in batch}

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        completed_count += 1

                        if completed_count % 10 == 0:
                            elapsed = time.time() - overall_start
                            rate = completed_count / elapsed if elapsed > 0 else 0
                            print(f"Progress: {completed_count}/{total_configs} ({rate:.2f} configs/sec)")

                            self.timing_logger.log_section(
                                f"{self.model_type}_batch_10",
                                10.0,
                                f"batch {completed_count // 10}"
                            )

                    except Exception as e:
                        print(f"Worker exception: {str(e)}")
                        print(traceback.format_exc())
        
        # Final summary
        overall_time = time.time() - overall_start
        self.timing_logger.log_total_elapsed()
        
        summary = self.progress_tracker.get_summary()
        
        print(f"\n{'='*80}")
        print(f"Tuning complete for {self.model_type}")
        print(f"Total time elapsed: {overall_time:.1f} seconds ({overall_time/60:.1f} minutes)")
        print(f"Configurations completed: {completed_count}")
        print(f"Summary: {summary}")
        print(f"Progress saved to: {self.progress_path}")
        print(f"Timing logs saved to: {self.timing_log_path}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    print("This is a base class. Extend it for specific model types (CRPS, MAPE, MSE).")
