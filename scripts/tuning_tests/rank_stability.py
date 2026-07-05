"""Rank-stability analysis for the repetition pilots.

Answers: "does a k-repetition ranking of hyperparameter configs agree with the
all-repetition ranking?" — the phase gate for running Phase-1 screening at 3 reps.

For each pilot shard CSV, configs are ranked by the mean of --metric over the
first k repetitions (run_num <= k-1... run_num is 1-based in the pilots, so
"first k" means the k lowest run_num values present). For every k from 1 to
max, that ranking is compared against the all-reps ranking via:

  * top-N overlap  — |top-N at k reps  ∩  top-N at all reps| / N
  * Spearman rho   — rank correlation over ALL configs in the shard

Method per Bouthillier et al. (MLSys 2021) and Dodge et al. (EMNLP 2019):
compare distributions/rankings across seeds, not single runs.

Depends only on the ProgressTracker CSV schema (run_num, config columns,
val_* metrics, status) — no imports from esb or the tuner package.

Usage (on the machine holding the results):
  python scripts/tuning_tests/rank_stability.py results/esb_tuner_scaled --shards 1 11 21 31
  python scripts/tuning_tests/rank_stability.py results/esb_tuner_scaled --shards 1 --metric val_mae --top-n 20
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from common import CONFIG_COLS, load_shard


def ranking(df: pd.DataFrame, metric: str, k: int | None) -> pd.Series:
    """Mean metric per config over the first k reps (all reps if k is None).

    Returns a Series indexed by config tuple, sorted best-first (lower = better
    for all val_* error metrics).
    """
    if k is not None:
        first_k = sorted(df["run_num"].unique())[:k]
        df = df[df["run_num"].isin(first_k)]
    return df.groupby(CONFIG_COLS)[metric].mean().sort_values()


def spearman(rank_a: pd.Series, rank_b: pd.Series) -> float:
    aligned = pd.concat([rank_a, rank_b], axis=1, keys=["a", "b"]).dropna()
    return aligned["a"].corr(aligned["b"], method="spearman")


def analyze_shard(df: pd.DataFrame, metric: str, top_n: int) -> pd.DataFrame:
    reps = sorted(df["run_num"].unique())
    full = ranking(df, metric, None)
    full_top = set(full.head(top_n).index)
    rows = []
    for k in range(1, len(reps) + 1):
        at_k = ranking(df, metric, k)
        k_top = set(at_k.head(top_n).index)
        rows.append({
            "reps": k,
            f"top{top_n}_overlap": len(k_top & full_top) / top_n,
            "spearman_all_configs": round(spearman(at_k, full), 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", type=Path, help="results folder holding the shard CSVs")
    p.add_argument("--shards", type=int, nargs="+", required=True,
                   help="pilot shard numbers, e.g. --shards 1 11 21 31")
    p.add_argument("--metric", default="val_mape",
                   help="ranking metric column (default: val_mape)")
    p.add_argument("--top-n", type=int, default=10,
                   help="size of the top set for overlap (default: 10)")
    p.add_argument("--out", type=Path, default=None,
                   help="optional CSV path for the combined stability table")
    args = p.parse_args()

    if not args.folder.is_dir():
        sys.exit(f"ERROR: {args.folder} is not a directory")

    combined = []
    for shard in args.shards:
        df = load_shard(args.folder, shard)
        lead_time = int(df["lead_time"].iloc[0])
        n_reps = df["run_num"].nunique()
        n_configs = df.groupby(CONFIG_COLS).ngroups
        table = analyze_shard(df, args.metric, args.top_n)
        table.insert(0, "lead_time", lead_time)
        table.insert(0, "shard", shard)
        combined.append(table)

        print(f"\n=== shard {shard}  (lead_time {lead_time}h, "
              f"{n_configs} configs, {n_reps} reps, metric {args.metric}) ===")
        print(table.to_string(index=False))

    result = pd.concat(combined, ignore_index=True)
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"\nSaved combined table -> {args.out}")

    # Verdict line: the smallest k where every shard clears both bars.
    overlap_col = f"top{args.top_n}_overlap"
    for k in sorted(result["reps"].unique()):
        at_k = result[result["reps"] == k]
        if len(at_k) == len(args.shards) and \
           (at_k[overlap_col] >= 0.7).all() and (at_k["spearman_all_configs"] >= 0.9).all():
            print(f"\nVERDICT: {k} rep(s) suffice — all shards have "
                  f">=70% top-{args.top_n} overlap and Spearman >=0.9 vs the full ranking.")
            break
    else:
        print("\nVERDICT: no rep count below the maximum clears the 70% overlap / "
              "0.9 Spearman bars on every shard — consider more Phase-1 reps.")


if __name__ == "__main__":
    main()
