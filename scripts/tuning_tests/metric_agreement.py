"""Metric-agreement test: do two ranking metrics pick the same top configs?

Ranks configs by the all-reps mean of each metric and reports, per shard, the
top-N set overlap and the full-ranking Spearman between the two. High overlap
= the ranking-metric choice is a non-issue for the Phase-2 cut; low overlap =
a real decision to settle BEFORE Phase 2 (the cut is where it becomes
irreversible).

Context: val_mape is a relative error (a 1°C miss counts more at 8°C than at
25°C — effectively low-temperature-weighted, which matches the cold-stunning
mission); val_mae weighs all temperatures equally. Disagreement between them
is a genuine difference in question asked, not a bug.

Usage (on the machine holding the results):
  python scripts/tuning_tests/metric_agreement.py results/esb_tuner_scaled --shards 1 11 21 31
  python scripts/tuning_tests/metric_agreement.py results/esb_tuner_scaled --shards 1 --metrics val_mape val_mae12
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from common import CONFIG_COLS, load_shard, mean_ranking


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", type=Path, help="results folder holding the shard CSVs")
    p.add_argument("--shards", type=int, nargs="+", required=True)
    p.add_argument("--metrics", nargs=2, default=["val_mape", "val_mae"],
                   help="the two ranking metrics to compare (default: val_mape val_mae)")
    p.add_argument("--top-n", type=int, default=20,
                   help="top-set size for the overlap (default: 20, the Phase-2 cut)")
    args = p.parse_args()

    if not args.folder.is_dir():
        sys.exit(f"ERROR: {args.folder} is not a directory")

    m_a, m_b = args.metrics
    for shard in args.shards:
        df = load_shard(args.folder, shard)
        lead_time = int(df["lead_time"].iloc[0])
        rank_a = mean_ranking(df, m_a)
        rank_b = mean_ranking(df, m_b)
        top_a = set(rank_a.head(args.top_n).index)
        top_b = set(rank_b.head(args.top_n).index)
        overlap = len(top_a & top_b) / args.top_n
        rho = pd.concat([rank_a, rank_b], axis=1, keys=["a", "b"]) \
                .corr(method="spearman").loc["a", "b"]
        print(f"shard {shard:3d} (LT {lead_time:3d}h): "
              f"top-{args.top_n} overlap {m_a} vs {m_b} = {overlap:.0%}, "
              f"Spearman = {rho:.3f}")

        only_a = top_a - top_b
        if only_a:
            print(f"  in top-{args.top_n} by {m_a} only: "
                  f"{sorted(only_a)[:5]}{' ...' if len(only_a) > 5 else ''}")


if __name__ == "__main__":
    main()
