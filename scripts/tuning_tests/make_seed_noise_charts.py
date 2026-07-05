"""Charts for the seed-noise test: every hyperparameter, one command.

From a seed_noise_by_group --out CSV, writes PNGs:
  * box_<param>.png    — box plot of rel_std by group, one per hyperparameter
  * scatter_<param>.png — mean vs std scatter, one color per group value

Replaces the by-hand Excel workflow (Excel PivotCharts cannot do XY scatter,
and manual series-per-group setup doesn't scale to 5 hyperparameters).

Usage (analysis env with pandas + matplotlib):
  python scripts/tuning_tests/make_seed_noise_charts.py results/tuning_tests_phase1/seed_noise_by_group.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PARAMS = ["activation", "num_layers", "neurons", "dropout", "cycle"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv", type=Path, help="seed_noise_by_group --out CSV")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="directory for PNGs (default: <csv folder>/charts)")
    p.add_argument("--log-std", action="store_true",
                   help="log-scale the std axis (useful when small nets dwarf large)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = args.out_dir or args.csv.parent / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    params = [c for c in PARAMS if c in df.columns and df[c].nunique() > 1]
    if not params:
        sys.exit("ERROR: no hyperparameter column with >1 value found")

    for param in params:
        groups = sorted(df[param].unique())

        # Box plot of relative noise by group value.
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.boxplot([df.loc[df[param] == g, "rel_std"] for g in groups],
                   tick_labels=[str(g) for g in groups])
        ax.set_xlabel(param)
        ax.set_ylabel("seed noise (std relative to shard median)")
        ax.set_title(f"Seed-noise distribution by {param}")
        if args.log_std:
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(out_dir / f"box_{param}.png", dpi=150)
        plt.close(fig)

        # Scatter: config quality (mean) vs config noise (std), colored by group.
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for g in groups:
            sub = df[df[param] == g]
            ax.scatter(sub["mean"], sub["std"], s=18, alpha=0.7, label=str(g))
        ax.set_xlabel("mean val metric (lower = better)")
        ax.set_ylabel("std across reps (lower = steadier)")
        ax.set_title(f"Config quality vs seed noise, by {param}")
        if args.log_std:
            ax.set_yscale("log")
        ax.legend(title=param, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_{param}.png", dpi=150)
        plt.close(fig)

    print(f"Wrote {2 * len(params)} chart(s) for {', '.join(params)} -> {out_dir}")


if __name__ == "__main__":
    main()
