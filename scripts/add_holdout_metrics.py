"""Compute holdout MAE / MAE<12 per model from a wide prediction CSV and write
them back into the campaign's progress CSV as new columns.

Why this exists
---------------
The progress CSV's metrics (``val_mae``, ``mae_2021``, ...) are written at
TRAINING time, so a dataset that did not exist then has no columns. Phase 5 also
TRAINED on 2021, which makes ``mae_2021`` in-sample and useless for picking a
rep. This recomputes the same two metrics against an arbitrary wide file.

Nothing in the viz needs changing: the explorer auto-discovers any numeric
column in the progress CSV, so the new columns appear in its dropdowns.

Metric definitions match src/evaluations/evaluation_functions.py:
  mae   = mean |pred - obs|
  mae12 = the same, restricted to rows where the OBSERVATION is < 12
          (the mask is on y_true, not on the prediction).

Ragged coverage is expected: models of different cycles cover different date
ranges and the wide file outer-joins them, so each model is scored only on the
rows where it actually predicted. The per-model row count is reported so a
model scored on a thin slice is visible rather than silently comparable.

Usage:
  python scripts/add_holdout_metrics.py \
      --progress results/phase5_final/mape_progress.csv \
      --wide     results/phase5_final/predictions_wide/esb_2025_2026_wide.csv \
      --suffix   2026
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

COLD_STUNNING_THRESHOLD = 12.0


def _fmt_lead_time(value) -> str:
    """3.0 -> '3', 4.5 -> '4.5'. Matches job_base_name's f-string on an int."""
    f = float(value)
    return str(int(f)) if f.is_integer() else f"{f:g}"


def build_name(row) -> str:
    """Rebuild the wide-CSV column name for one progress row.

    Deliberately mirrors tuner_utils.job_base_name rather than parsing the
    column string: the grid axes are the source of truth, and a parser would
    have to guess where an activation ends and a layer count begins.
    """
    return (
        f"{row['model_type']}_{_fmt_lead_time(row['lead_time'])}h"
        f"_cycle{row['cycle']}_{row['activation']}"
        f"_{row['num_layers']}L_{row['neurons']}N"
        f"_d{row.get('dropout', 0.0)}"
        f"_run{row['run_num']}"
    )


def score(obs: pd.Series, pred: pd.Series):
    """Return (mae, mae12, n_rows, n_cold) over the rows both series cover."""
    both = pd.concat([obs, pred], axis=1).dropna()
    if both.empty:
        return np.nan, np.nan, 0, 0
    o = both.iloc[:, 0].to_numpy(dtype=float)
    p = both.iloc[:, 1].to_numpy(dtype=float)
    err = np.abs(p - o)
    cold = o < COLD_STUNNING_THRESHOLD
    mae12 = float(err[cold].mean()) if cold.any() else np.nan
    return float(err.mean()), mae12, len(o), int(cold.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--progress", type=Path, required=True,
                    help="campaign progress CSV; edited IN PLACE (a .bak is kept)")
    ap.add_argument("--wide", type=Path, required=True,
                    help="wide prediction CSV to score against")
    ap.add_argument("--suffix", required=True,
                    help="column suffix, e.g. 2026 -> mae_2026, mae12_2026")
    ap.add_argument("--dry-run", action="store_true",
                    help="report coverage and stop without writing")
    args = ap.parse_args()

    for path in (args.progress, args.wide):
        if not path.is_file():
            print(f"ERROR: not found: {path}")
            return 2

    progress = pd.read_csv(args.progress)
    wide = pd.read_csv(args.wide, index_col=0)
    if "observation" not in wide.columns:
        print(f"ERROR: {args.wide} has no 'observation' column")
        return 2
    obs = wide["observation"]

    mae_col = f"mae_{args.suffix}"
    mae12_col = f"mae12_{args.suffix}"

    maes, mae12s, matched = [], [], 0
    thin, missing = [], []
    for _, row in progress.iterrows():
        name = build_name(row)
        if name not in wide.columns:
            missing.append(name)
            maes.append(np.nan)
            mae12s.append(np.nan)
            continue
        matched += 1
        m, m12, n, n_cold = score(obs, wide[name])
        if n and n < 0.5 * len(wide):
            thin.append((name, n))
        if n_cold == 0:
            # An all-NaN mae12 column would otherwise look like a bug in the
            # explorer rather than "this year never got cold".
            pass
        maes.append(m)
        mae12s.append(m12)

    progress[mae_col] = maes
    progress[mae12_col] = mae12s

    print(f"Scored {matched}/{len(progress)} progress rows against {args.wide.name}")
    print(f"  rows in wide file: {len(wide)}")
    n_cold_total = int((obs.dropna() < COLD_STUNNING_THRESHOLD).sum())
    print(f"  observations < {COLD_STUNNING_THRESHOLD:g}C: {n_cold_total}")
    if n_cold_total == 0:
        print(f"  WARNING: no cold-stunning rows, so {mae12_col} is ALL NaN. "
              f"That is a property of the year, not a failure.")
    if missing:
        print(f"  WARNING: {len(missing)} progress rows had no column in the "
              f"wide file (not predicted). First few: {missing[:3]}")
    if thin:
        print(f"  NOTE: {len(thin)} models covered <50% of rows (ragged cycles). "
              f"First few: {thin[:3]}")
    if matched == 0:
        print("ERROR: nothing matched — the name format likely disagrees. "
              "Compare a progress row against a wide column header.")
        return 1

    if args.dry_run:
        print("--dry-run: not written.")
        return 0

    # Copy the ORIGINAL bytes, before the in-memory frame (which now carries the
    # new columns) is written over the file. Never overwrite an existing .bak:
    # re-running must not destroy the pre-first-run state.
    import shutil
    backup = args.progress.with_suffix(args.progress.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(args.progress, backup)
    progress.to_csv(args.progress, index=False)
    print(f"Wrote {mae_col}, {mae12_col} -> {args.progress}"
          + (f" (backup: {backup.name})" if backup.exists() else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
