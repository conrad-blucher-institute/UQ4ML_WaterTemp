"""Shared helpers for the tuning-test scripts (one test = one script).

Everything here depends ONLY on the ProgressTracker CSV schema
(esb_03_2026_tune_experiment/tuner_utils.py FIELDNAMES) — never on esb or
tuner imports. That contract is what keeps these scripts portable and makes
an eventual promotion into `esb analyze` cheap.
"""

import sys
from pathlib import Path

import pandas as pd

CONFIG_COLS = ["cycle", "activation", "num_layers", "neurons", "dropout"]


def load_shard(folder: Path, shard: int, single_lead_time: bool = True) -> pd.DataFrame:
    """Load one shard's progress CSV, completed rows only. Loud on ambiguity."""
    matches = sorted(folder.glob(f"*progress_shard{shard}of*.csv"))
    if len(matches) != 1:
        sys.exit(f"ERROR: expected exactly 1 CSV for shard {shard} in {folder}, "
                 f"found {len(matches)}: {[m.name for m in matches]}")
    df = pd.read_csv(matches[0])
    df = df[df["status"].str.lower().eq("completed")].copy()
    if df.empty:
        sys.exit(f"ERROR: {matches[0].name} has no completed rows")
    if single_lead_time:
        lead_times = df["lead_time"].unique()
        if len(lead_times) != 1:
            sys.exit(f"ERROR: {matches[0].name} spans multiple lead times "
                     f"{lead_times} — pilot analyses assume one per shard")
    return df


def mean_ranking(df: pd.DataFrame, metric: str) -> pd.Series:
    """Mean metric per config over all reps present, sorted best-first
    (lower = better for the val_* error metrics)."""
    return df.groupby(CONFIG_COLS)[metric].mean().sort_values()
