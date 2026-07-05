"""Charts for the rank-stability test: stability curves as PNGs.

From a rank_stability --out CSV (long format: one row per shard x reps),
writes one PNG per metric column (top-N overlap, Spearman) with a line per
shard. Shards are labeled "shard K (LT Xh)" so several shards on the same
lead time (e.g. 21/23/28 all LT96) stay distinguishable — the old xlsx
pivot-on-lead_time broke on exactly that.

The verdict bars from rank_stability (70% overlap / 0.9 Spearman) are drawn
as dashed reference lines so the sufficient rep count reads straight off
the chart.

Usage (analysis env with pandas + matplotlib):
  python scripts/tuning_tests/make_stability_charts.py results/tuning_tests_phase1/rank_stability.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ID_COLS = {"shard", "lead_time", "reps"}
# axis floor + verdict bar, matched by metric-column-name prefix
Y_STYLE = {"top": (0.4, 0.7), "spearman": (0.6, 0.9)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv", type=Path, help="rank_stability --out CSV")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="directory for PNGs (default: <csv folder>/charts)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    metrics = [c for c in df.columns if c not in ID_COLS]
    if not metrics:
        sys.exit(f"ERROR: no metric columns found in {args.csv}")
    out_dir = args.out_dir or args.csv.parent / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(df["shard"].unique())
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for shard in shards:
            sub = df[df["shard"] == shard].sort_values("reps")
            lt = int(sub["lead_time"].iloc[0])
            ax.plot(sub["reps"], sub[metric], marker="o", markersize=4,
                    label=f"shard {shard} (LT {lt}h)")

        y_min, bar = next((v for pre, v in Y_STYLE.items()
                           if metric.startswith(pre)), (None, None))
        if y_min is not None:
            ax.set_ylim(y_min, 1.02)
            ax.axhline(bar, linestyle="--", linewidth=1, color="gray",
                       label=f"verdict bar ({bar})")
        ax.set_xticks(sorted(df["reps"].unique()))
        ax.set_xlabel("repetitions k")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs repetitions (k-rep ranking vs all-rep reference)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"stability_{metric}.png", dpi=150)
        plt.close(fig)

    print(f"Wrote {len(metrics)} chart(s) for {', '.join(metrics)} -> {out_dir}")


if __name__ == "__main__":
    main()
