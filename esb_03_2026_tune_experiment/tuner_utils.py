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
    
    def __init__(self, progress_path: Path):
        self.progress_path = progress_path
        self.completed = set()
        self.results = []
        
        # Load existing progress if file exists
        if progress_path.exists():
            with open(progress_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    config_key = self._make_key(row)
                    self.completed.add(config_key)
                    self.results.append(row)
        else:
            # Create CSV with headers (include run_num so multiple runs append)
            with open(progress_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'run_num', 'model_type', 'lead_time', 'cycle', 'activation', 'num_layers',
                    'neurons', 'loss_value', 'val_mae', 'val_mae12', 'mae_2021', 'mae12_2021',
                    'metrics', 'timestamp', 'status'
                ])
                writer.writeheader()
    
    @staticmethod
    def _make_key(config: Dict[str, Any]) -> str:
        """Create a unique key from a configuration."""
        return f"{config['model_type']}_{config['lead_time']}_{config['cycle']}_" \
               f"{config['activation']}_{config['num_layers']}_{config['neurons']}"
    
    def is_completed(self, model_type: str, lead_time: int, cycle: int,
                     activation: str, num_layers: int, neurons: int, run_num: int = 0) -> bool:
        """Check if a configuration has been completed successfully."""
        # Only consider rows that have status 'completed' (skip errored rows so they get retried)
        if self.results:
            for row in self.results:
                try:
                    status = str(row.get('status', '')).lower()
                    if status.startswith('error') or status == '':
                        continue
                    if (row['model_type'] == str(model_type) and int(row['lead_time']) == int(lead_time)
                        and int(row['cycle']) == int(cycle) and row['activation'] == str(activation)
                        and int(row['num_layers']) == int(num_layers) and int(row['neurons']) == int(neurons)
                        and int(row.get('run_num', 0)) == int(run_num)):
                        return True
                except Exception:
                    continue
        return False
    
    def log_result(self, model_type: str, lead_time: int, cycle: int,
                   activation: str, num_layers: int, neurons: int,
                   loss_value: float, status: str = "completed", run_num: int = 0, metrics: Dict[str, Any] = None):
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
            'loss_value': loss_value,
            'val_mae': _extract('val_mae'),
            'val_mae12': _extract('val_mae12'),
            'mae_2021': _extract('mae_2021'),
            'mae12_2021': _extract('mae12_2021'),
            'metrics': '' if not metrics else json.dumps(metrics),
            'timestamp': datetime.now().isoformat(),
            'status': status
        }

        key = self._make_key(config)
        self.completed.add(key)
        self.results.append(config)

        # Append to CSV
        with open(self.progress_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'run_num', 'model_type', 'lead_time', 'cycle', 'activation', 'num_layers',
                'neurons', 'loss_value', 'val_mae', 'val_mae12', 'mae_2021', 'mae12_2021',
                'metrics', 'timestamp', 'status'
            ])
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


class GridSearchConfig:
    """Defines and generates grid search configurations."""
    
    def __init__(self, model_type: str, lead_times: List[int], cycles: List[int], activations: List[str], num_layers_range: List[int], neurons_range: List[int], run_nums: List[int] = None):

        self.model_type = model_type
        self.lead_times = lead_times
        self.cycles = cycles
        self.activations = activations
        self.num_layers_range = num_layers_range
        self.neurons_range = neurons_range
        # allow generating multiple independent runs per configuration
        self.run_nums = [0] if run_nums is None else list(run_nums)
    
    def generate_configs(self) -> List[Dict[str, Any]]:
        """Generate all configurations for grid search."""
        configs = []
        for lead_time in self.lead_times:
            for cycle in self.cycles:
                for activation in self.activations:
                    for num_layers in self.num_layers_range:
                        for neurons in self.neurons_range:
                            for run_num in self.run_nums:
                                configs.append({
                                    'run_num': run_num,
                                    'model_type': self.model_type,
                                    'lead_time': lead_time,
                                    'cycle': cycle,
                                    'activation': activation,
                                    'num_layers': num_layers,
                                    'neurons': neurons
                                })
        return configs
    
    def count_configs(self) -> int:
        """Return total number of configurations in grid."""
        return (len(self.lead_times) * len(self.cycles) * len(self.activations) * 
            len(self.num_layers_range) * len(self.neurons_range) * len(self.run_nums))
