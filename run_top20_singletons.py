"""Run the top-20 combos as SINGLETON esb grid runs (one job per combo).

The esb CLI drives a cartesian-product grid; the top-20 combos are arbitrary
points, so we express each combo as an all-singleton grid (1 job) and let
--repetitions multiply it.

PARALLELISM: within one esb run the reps loop is sequential and the grid is a
single config, so --workers gives no speedup. The parallelism lever is running
whole COMBOS concurrently (--parallel N). To keep that race-free, each combo
gets its OWN output subdir (mirrors the --shard "no shared mutable files"
design), so N concurrent runs never touch the same progress CSV / provenance.

    results/top20_singletons/
        MAPE_12h_cycle0_relu_1L_16N_d0.0/   <- combo 1: mape_progress.csv, keras_files/, run_provenance.json, console.log
        MAPE_12h_cycle0_relu_1L_256N_d0.1/  <- combo 2
        ...

Merge for analysis later: concat the per-combo mape_progress.csv files (a plain
union — same pattern as get_top_20_csvs.py).

Tack on more reps later: re-run with a larger --repetitions and the SAME
--output_dir; each combo's resume-filter skips done run_nums, only new ones train.

Shells out to `python -m esb run` (no esb imports — obeys the scripts rule).
Prints the plan by default; pass --launch to actually execute.

    # preview the 12h plan (no training)
    python run_top20_singletons.py --lead_times 12 --repetitions 10 --parallel 4

    # run it, 4 combos at a time
    python run_top20_singletons.py --lead_times 12 --repetitions 10 --parallel 4 --launch
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

COMBO_COLS = ["lead_time", "cycle", "activation", "num_layers", "neurons", "dropout"]


def combo_sig(r) -> str:
    """Per-combo dir/name (matches the .keras base-name convention, no run_num)."""
    return (f"MAPE_{int(r.lead_time)}h_cycle{int(r.cycle)}_{r.activation}"
            f"_{int(r.num_layers)}L_{int(r.neurons)}N_d{float(r.dropout)}")


def build_cmd(r, args, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable, "-m", "esb", "run",
        "--loss", "mape",
        "--lead_times", str(int(r.lead_time)),
        "--rotations", str(int(r.cycle)),
        "--activations", str(r.activation),
        "--num_layers", str(int(r.num_layers)),
        "--neurons", str(int(r.neurons)),
        "--dropout", str(float(r.dropout)),
        "--repetitions", str(args.repetitions),
        "--workers", str(args.workers),
        "--output_dir", str(out_dir),
    ]
    if args.scale:
        cmd.append("--scale")
    return cmd


def run_one(idx: int, total: int, r, args) -> int:
    out_dir = Path(args.output_dir) / combo_sig(r)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(r, args, out_dir)
    log = out_dir / "console.log"
    # Force UTF-8 stdio in the child AND its spawned worker processes. Without
    # this, Windows cp1252 stdout can't encode the '->' (U+2192) that the data-
    # prep code prints, and every worker dies before logging a result. Env is
    # inherited by the ProcessPoolExecutor workers, so this reaches them too.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    with open(log, "w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
    print(f"[{idx}/{total}] {'ok ' if rc == 0 else f'EXIT {rc}'}  {combo_sig(r)}  (log: {log})")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combos-file", default="data/top_20_combos.csv")
    ap.add_argument("--lead_times", type=int, nargs="+", default=[12],
                    help="Only run combos whose lead_time is in this set.")
    ap.add_argument("--repetitions", type=int, default=10)
    ap.add_argument("--output_dir", default="results/top20_singletons")
    ap.add_argument("--parallel", type=int, default=1,
                    help="How many combos to run at once (~= cores to use). "
                         "Each combo is ~1 core, so set this near your core count.")
    ap.add_argument("--workers", type=int, default=1,
                    help="esb workers per combo. Moot for singleton runs (reps are "
                         "sequential, grid is 1 config), so leave at 1.")
    ap.add_argument("--scale", action="store_true", default=True,
                    help="Fit StandardScaler on train only (matches the scaled campaign).")
    ap.add_argument("--no-scale", dest="scale", action="store_false")
    ap.add_argument("--launch", action="store_true",
                    help="Actually execute. Without it, only prints the plan.")
    args = ap.parse_args()

    df = pd.read_csv(args.combos_file)
    df = df[df["lead_time"].isin(args.lead_times)]
    combos = df[COMBO_COLS].drop_duplicates().sort_values(COMBO_COLS)
    if combos.empty:
        print(f"No combos in {args.combos_file} for lead_times={args.lead_times}")
        return 1
    rows = [r for _, r in combos.iterrows()]
    total = len(rows)

    print(f"lead_times={args.lead_times}: {total} distinct combos "
          f"x {args.repetitions} reps = {total * args.repetitions} trainings")
    print(f"output_dir={args.output_dir}  scale={args.scale}  "
          f"parallel={args.parallel}  workers={args.workers}")
    print(f"mode={'LAUNCH' if args.launch else 'PRINT-ONLY (pass --launch to run)'}\n")

    if not args.launch:
        for i, r in enumerate(rows, 1):
            out_dir = Path(args.output_dir) / combo_sig(r)
            print(f"[{i}/{total}] {' '.join(build_cmd(r, args, out_dir)[2:])}")
        return 0

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futs = {pool.submit(run_one, i, total, r, args): i
                for i, r in enumerate(rows, 1)}
        for fut in as_completed(futs):
            if fut.result() != 0:
                failures += 1
    print(f"\nDone. {total - failures}/{total} combos ok, {failures} failed "
          f"(see each combo's console.log).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
