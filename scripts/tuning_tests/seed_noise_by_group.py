"""Seed-noise heterogeneity test: is seed noise uniform across the grid?

Per-config seed noise = std of --metric across repetitions of the SAME config.
If it is roughly the same size across activations (and other hyperparameter
groups), the rank-stability result from activation-biased pilot shards
generalizes grid-wide; if one group is much noisier, buy ONE targeted pilot
shard covering that slice instead of several.

The pooled table normalizes each shard's stds by that shard's median, so lead
times with different absolute metric scales are comparable — read it as
"how many times noisier than typical is this group?".

Usage (on the machine holding the results):
  python scripts/tuning_tests/seed_noise_by_group.py results/esb_tuner_scaled --shards 1 11 21 31
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from common import CONFIG_COLS, load_shard

GROUP_COLS = ["activation", "num_layers", "neurons", "dropout", "cycle"]


def seed_noise(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per config: mean, std (over reps), and rep count of the metric."""
    per_config = df.groupby(CONFIG_COLS)[metric].agg(["mean", "std", "count"])
    if (per_config["count"] < 2).any():
        n = int((per_config["count"] < 2).sum())
        print(f"  note: {n} config(s) have <2 reps — std undefined, excluded")
    return per_config.dropna(subset=["std"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", type=Path, help="results folder holding the shard CSVs")
    p.add_argument("--shards", type=int, nargs="+", required=True)
    p.add_argument("--metric", default="val_mape",
                   help="metric for the seed-noise breakdown (default: val_mape)")
    args = p.parse_args()

    if not args.folder.is_dir():
        sys.exit(f"ERROR: {args.folder} is not a directory")

    all_noise = []
    for shard in args.shards:
        df = load_shard(args.folder, shard)
        lead_time = int(df["lead_time"].iloc[0])
        print(f"\n=== shard {shard} (lead_time {lead_time}h, "
              f"{df.groupby(CONFIG_COLS).ngroups} configs, "
              f"{df['run_num'].nunique()} reps) ===")

        noise = seed_noise(df, args.metric).reset_index()
        noise["shard"], noise["lead_time"] = shard, lead_time
        all_noise.append(noise)

        print(f"\n  Seed noise of {args.metric} (median per-config std, "
              f"by group; n = configs in group):")
        for col in GROUP_COLS:
            summary = noise.groupby(col)["std"].agg(["median", "count"])
            cells = "  ".join(f"{v}: {row['median']:.3f} (n={int(row['count'])})"
                              for v, row in summary.iterrows())
            print(f"    {col:12s} {cells}")

    # Cross-shard pooled view: is any group consistently noisier?
    pooled = pd.concat(all_noise, ignore_index=True)
    pooled["rel_std"] = pooled["std"] / pooled.groupby("shard")["std"].transform("median")
    print("\n=== pooled across shards (std relative to each shard's median) ===")
    for col in GROUP_COLS:
        summary = pooled.groupby(col)["rel_std"].agg(["median", "count"])
        cells = "  ".join(f"{v}: {row['median']:.2f}x (n={int(row['count'])})"
                          for v, row in summary.iterrows())
        print(f"  {col:12s} {cells}")
    print("\nReading guide: ~1.0x everywhere = homogeneous noise (rank-stability "
          "result generalizes). A group at >2x = noisy slice; consider one "
          "targeted pilot shard covering it.")


if __name__ == "__main__":
    main()
