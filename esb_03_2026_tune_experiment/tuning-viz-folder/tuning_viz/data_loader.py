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
                          E.g., 'val_mae', 'val_mape', 'val_loss'
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
        df = pd.read_csv(csv_path, header=None)
        self.raw_data = df
        
        # Parse the DataFrame
        # Expected structure based on mape_progress.csv:
        # [0]=run_id, [1]=metric_type, [2]=leadtime, [3]=cycle, 
        # [4]=activation, [5]=num_layers, [6]=hidden_units, [7]=best_val_metric,
        # [8]=metrics_dict_json, [9]=timestamp, [10]=status
        
        parsed = pd.DataFrame()
        parsed['run_id'] = df[0]
        parsed['metric_type'] = df[1]
        parsed['leadtime'] = df[2]
        parsed['cycle'] = df[3]
        parsed['activation'] = df[4]
        parsed['num_layers'] = df[5]
        parsed['neurons'] = df[6]
        parsed['best_val_metric'] = df[7]
        parsed['timestamp'] = df[9]
        parsed['status'] = df[10]
        parsed['iteration'] = iteration_id

        # Ensure numeric columns are properly typed
        numeric_cols = ['leadtime', 'cycle', 'num_layers', 'neurons', 'best_val_metric']
        for col in numeric_cols:
            parsed[col] = pd.to_numeric(parsed[col], errors='coerce')
        
        # Extract metrics from JSON dict
        metrics_dict_list = []
        for idx, row in df.iterrows():
            try:
                metrics = json.loads(row[8])
                metrics_dict_list.append(metrics)
            except (json.JSONDecodeError, TypeError):
                metrics_dict_list.append({})
        
        metrics_df = pd.DataFrame(metrics_dict_list)
        
        # Merge metrics into parsed dataframe
        for col in metrics_df.columns:
            parsed[col] = metrics_df[col]
        
        # Ensure metric column exists
        if self.metric_column not in parsed.columns:
            available = [c for c in parsed.columns if 'val_' in c or 'loss' in c]
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
