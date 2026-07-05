"""
Shared utilities for hyperparameter tuning with grid search, progress tracking, and time logging.
"""

import csv
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple


class TimingLogger:
    """Logs timing information at component and section levels to CSV."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.start_time = time.time()
        self.section_times: Dict[str, List[float]] = {}
        self.component_times: List[Tuple[str, float]] = []
        
        # Initialize CSV with headers
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'event_type', 'component', 'section', 'duration_sec', 'notes'])
    
    def log_component(self, component_name: str, duration_sec: float, notes: str = ""):
        """Log a component-level timing."""
        timestamp = datetime.now().isoformat()
        self.component_times.append((component_name, duration_sec))
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, 'component', component_name, '', duration_sec, notes])
    
    def log_section(self, section_name: str, duration_sec: float, notes: str = ""):
        """Log a section-level timing."""
        timestamp = datetime.now().isoformat()
        
        if section_name not in self.section_times:
            self.section_times[section_name] = []
        self.section_times[section_name].append(duration_sec)
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, 'section', '', section_name, duration_sec, notes])
    
    def log_total_elapsed(self):
        """Log total elapsed time."""
        total = time.time() - self.start_time
        timestamp = datetime.now().isoformat()
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, 'total', '', '', total, 'script complete'])


class ProgressTracker:
    """Tracks tuning progress and allows checkpointing/resumption."""

    # Single source of truth for the progress CSV schema. Stage C additions:
    #  * 'dropout' — part of the resume identity (each dropout rate is its own job)
    #  * the FULL metric suite promoted to columns (mae/mae12/me/me12/mape on both
    #    the validation set and the 2021 independent year) so any of them can be
    #    inspected in the viz suite without re-parsing the JSON 'metrics' blob.
    FIELDNAMES = [
        'run_num', 'model_type', 'lead_time', 'cycle', 'activation', 'num_layers',
        'neurons', 'dropout', 'loss_value',
        'val_mae', 'val_mae12', 'val_me', 'val_me12', 'val_mape',
        'mae_2021', 'mae12_2021', 'me_2021', 'me12_2021', 'mape_2021',
        'metrics', 'timestamp', 'status', 'duration_sec',
    ]
    # Promoted metric column -> key looked up in the metrics dict.
    _PROMOTED_METRICS = [
        'val_mae', 'val_mae12', 'val_me', 'val_me12', 'val_mape',
        'mae_2021', 'mae12_2021', 'me_2021', 'me12_2021', 'mape_2021',
    ]

    def __init__(self, progress_path: Path):
        self.progress_path = progress_path
        self.completed = set()
        self.results = []

        # Load existing progress if file exists
        if progress_path.exists():
            with open(progress_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                header = reader.fieldnames or []
                for row in reader:
                    self.results.append(row)
                    status = str(row.get('status', '')).lower()
                    if not status.startswith('error') and status != '':
                        config_key = self._make_key(row)
                        self.completed.add(config_key)
            # Schema migration: if the CSV predates a FIELDNAMES addition (e.g.
            # duration_sec), rewrite it under the current header so appended rows
            # and the header always agree (a bare append would desync them).
            if header != self.FIELDNAMES:
                with open(progress_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                    writer.writeheader()
                    for row in self.results:
                        writer.writerow({k: row.get(k, '') for k in self.FIELDNAMES})
        else:
            # Create CSV with headers (include run_num so multiple runs append)
            with open(progress_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    @staticmethod
    def _make_key(config: Dict[str, Any]) -> str:
        """Create a unique key from a configuration (includes dropout + run_num).

        Reading legacy rows without a 'dropout' column falls back to 0.0 so old
        progress CSVs still resolve to a stable key.
        """
        run_num = config.get('run_num', 0)
        dropout = config.get('dropout', 0.0)
        return f"{config['model_type']}_{config['lead_time']}_{config['cycle']}_" \
               f"{config['activation']}_{config['num_layers']}_{config['neurons']}_" \
               f"{dropout}_{run_num}"

    def is_completed(self, model_type: str, lead_time: int, cycle: int,
                     activation: str, num_layers: int, neurons: int,
                     dropout: float = 0.0, run_num: int = 0) -> bool:
        """Check if a configuration has been completed successfully (O(1) set lookup)."""
        key = f"{model_type}_{lead_time}_{cycle}_{activation}_{num_layers}_" \
              f"{neurons}_{dropout}_{run_num}"
        return key in self.completed

    def log_result(self, model_type: str, lead_time: int, cycle: int,
                   activation: str, num_layers: int, neurons: int,
                   loss_value: float, status: str = "completed", run_num: int = 0,
                   metrics: Dict[str, Any] = None, dropout: float = 0.0,
                   duration_sec: float = None):
        """Log a tuning result."""
        # Promote key metrics to top-level columns so the CSV is queryable without JSON parsing
        def _extract(key):
            if not metrics:
                return ''
            v = metrics.get(key, '')
            return float(v) if v != '' else ''

        config = {
            'run_num': run_num,
            'model_type': model_type,
            'lead_time': lead_time,
            'cycle': cycle,
            'activation': activation,
            'num_layers': num_layers,
            'neurons': neurons,
            'dropout': dropout,
            'loss_value': loss_value,
            'metrics': '' if not metrics else json.dumps(metrics, default=lambda o: float(o) if hasattr(o, 'item') else str(o)),
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'duration_sec': '' if duration_sec is None else round(float(duration_sec), 1),
        }
        # Fill every promoted metric column from the metrics dict.
        for col in self._PROMOTED_METRICS:
            config[col] = _extract(col)

        key = self._make_key(config)
        if not status.lower().startswith('error') and status != '':
            self.completed.add(key)
        self.results.append(config)

        # Append to CSV
        with open(self.progress_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(config)

    def aggregate_runs(self, aggregate_path: Path):
        """Aggregate multiple runs into a single CSV with mean/std/count per config."""
        try:
            import pandas as pd
        except Exception:
            return None

        if not self.progress_path.exists():
            return None
        df = pd.read_csv(self.progress_path)
        if df.empty:
            return None

        group_cols = ['model_type', 'lead_time', 'cycle', 'activation', 'num_layers', 'neurons']
        agg = df.groupby(group_cols).agg(
            runs=('run_num', 'nunique'),
            mean_loss=('loss_value', 'mean'),
            std_loss=('loss_value', 'std')
        ).reset_index()

        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(aggregate_path, index=False)
        return aggregate_path
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of progress."""
        return {
            'total_completed': len(self.completed),
            'total_results': len(self.results),
            'by_model': self._count_by_model()
        }
    
    def _count_by_model(self) -> Dict[str, int]:
        """Count completed configurations by model type."""
        counts = {}
        for result in self.results:
            model = result['model_type']
            counts[model] = counts.get(model, 0) + 1
        return counts


def job_base_name(config: Dict[str, Any]) -> str:
    """Filesystem base name for one job's artifacts (.keras / _history.json /
    _scaler.joblib). SINGLE definition so every artifact of a job shares one
    prefix and no two jobs can collide.

    Includes EVERY grid axis. Dropout was historically missing, so the 4 jobs
    differing only in dropout silently overwrote each other's saved model,
    history, and scaler (progress CSV rows stayed distinct — the loss looked
    fine while 3 of 4 models were gone).
    """
    return (
        f"{config['model_type']}_{config['lead_time']}h"
        f"_cycle{config['cycle']}_{config['activation']}"
        f"_{config['num_layers']}L_{config['neurons']}N"
        f"_d{config.get('dropout', 0.0)}"
        f"_run{config['run_num']}"
    )


class GridSearchConfig:
    """Defines and generates grid search configurations."""
    
    def __init__(self, model_type: str, lead_times: List[int], cycles: List[int], activations: List[str], num_layers_range: List[int], neurons_range: List[int], run_nums: List[int] = None, dropouts: List[float] = None):

        self.model_type = model_type
        self.lead_times = lead_times
        self.cycles = cycles
        self.activations = activations
        self.num_layers_range = num_layers_range
        self.neurons_range = neurons_range
        # allow generating multiple independent runs per configuration
        self.run_nums = [0] if run_nums is None else list(run_nums)
        # Dropout rate(s) applied after each hidden Dense (Stage C, design §9).
        # None -> [0.0] so the legacy hardcoded grids (which don't pass dropouts)
        # generate exactly one no-dropout config per point, unchanged.
        self.dropouts = [0.0] if dropouts is None else list(dropouts)

    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all configurations for grid search."""
        configs = []
        for lead_time in self.lead_times:
            for cycle in self.cycles:
                for activation in self.activations:
                    for num_layers in self.num_layers_range:
                        for neurons in self.neurons_range:
                            for dropout in self.dropouts:
                                for run_num in self.run_nums:
                                    configs.append({
                                        'run_num': run_num,
                                        'model_type': self.model_type,
                                        'lead_time': lead_time,
                                        'cycle': cycle,
                                        'activation': activation,
                                        'num_layers': num_layers,
                                        'neurons': neurons,
                                        'dropout': dropout
                                    })
        return configs

    def count_configs(self) -> int:
        """Return total number of configurations in grid."""
        return (len(self.lead_times) * len(self.cycles) * len(self.activations) *
            len(self.num_layers_range) * len(self.neurons_range) *
            len(self.dropouts) * len(self.run_nums))
