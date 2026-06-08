"""
Data loading and parsing for hyperparameter tuning results.
"""

import pandas as pd
import json
from typing import List


class TuningResultsLoader:
    """Load and parse hyperparameter tuning results from CSV files."""

    def __init__(self, metric_column: str = "val_mae"):
        """
        Args:
            metric_column: Which metric to use for visualization.
                          E.g., 'val_mae', 'val_mae12', 'mae_2021', 'mae12_2021'
        """
        self.metric_column = metric_column
        self.data = None
        self.raw_data = None

    def load(self, csv_path: str, iteration_id: int = 1) -> pd.DataFrame:
        """
        Load and parse a tuning results CSV.

        Args:
            csv_path: Path to CSV file
            iteration_id: Which iteration this is (for multi-run tracking)

        Returns:
            Parsed DataFrame with extracted metrics
        """
        # header=0 means "row 0 is the header row" — pandas uses those strings as
        # column names instead of assigning numeric indices 0, 1, 2, ...
        # This means adding/reordering columns in the CSV never breaks the loader.
        df = pd.read_csv(csv_path, header=0)
        self.raw_data = df

        # Rename columns to the names the visualizer expects internally
        rename_map = {
            'run_num':    'run_id',
            'model_type': 'metric_type',
            'lead_time':  'leadtime',
        }
        df = df.rename(columns=rename_map)

        parsed = df.copy()
        parsed['iteration'] = iteration_id

        # Ensure numeric columns are properly typed
        numeric_cols = ['leadtime', 'cycle', 'num_layers', 'neurons', 'loss_value',
                        'val_mae', 'val_mae12', 'mae_2021', 'mae12_2021']
        for col in numeric_cols:
            if col in parsed.columns:
                parsed[col] = pd.to_numeric(parsed[col], errors='coerce')

        # Extract any remaining metrics from the JSON blob in 'metrics' column.
        # The top-level columns (val_mae, val_mae12, mae_2021, mae12_2021) are
        # already present as direct columns, but any other history keys
        # (val_mse, val_mape, etc.) are still in the JSON and get merged here.
        if 'metrics' in parsed.columns:
            metrics_dict_list = []
            for raw_val in parsed['metrics']:
                try:
                    metrics_dict_list.append(json.loads(raw_val))
                except (json.JSONDecodeError, TypeError):
                    metrics_dict_list.append({})

            metrics_df = pd.DataFrame(metrics_dict_list, index=parsed.index)

            # Only add columns that aren't already top-level to avoid overwriting
            for col in metrics_df.columns:
                if col not in parsed.columns:
                    parsed[col] = metrics_df[col]

        # Ensure the requested metric column exists
        if self.metric_column not in parsed.columns:
            available = [c for c in parsed.columns
                         if any(k in c for k in ('val_', 'mae', 'mse', 'mape', 'loss', '2021'))]
            raise ValueError(
                f"Metric column '{self.metric_column}' not found. "
                f"Available metrics: {available}"
            )

        self.data = parsed
        return parsed

    def load_multiple(self, csv_paths: List[str]) -> pd.DataFrame:
        """
        Load multiple CSV files (from different iterations).

        Args:
            csv_paths: List of CSV file paths

        Returns:
            Combined DataFrame with iteration_id column
        """
        dfs = []
        for i, path in enumerate(csv_paths, start=1):
            df = self.load(path, iteration_id=i)
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        self.data = combined
        return combined
