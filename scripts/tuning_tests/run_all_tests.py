"""Run every tuning test against the same shards in one command.

Thin dispatcher only: each test stays its own script and stays runnable
individually; this just invokes them in sequence with shared arguments.
metric_agreement runs three times — one per metric pair, including both
mae12 combos (the long-lead-time cold-tail divergence recheck).

With --out-dir, the chart scripts run afterwards on the saved CSVs
(PNGs land in <out-dir>/charts). Pass --no-charts to skip them.

Usage:
  python scripts/tuning_tests/run_all_tests.py --shards 1 11 21 23 28 31 --max-reps 10 --out-dir results/tuning_tests_phase1
  (--results-dir defaults to results/esb_tuner_scaled; keyword-only, no positionals)
"""

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# (script, extra args, --out stem)
TESTS = [
    ("rank_stability.py", [], "rank_stability"),
    ("seed_noise_by_group.py", [], "seed_noise_by_group"),
    ("metric_agreement.py", ["--metrics", "val_mape", "val_mae"], "metric_agreement_mape_mae"),
    ("metric_agreement.py", ["--metrics", "val_mape", "val_mae12"], "metric_agreement_mape_mae12"),
    ("metric_agreement.py", ["--metrics", "val_mae", "val_mae12"], "metric_agreement_mae_mae12"),
]

# (chart script, --out stem of the CSV it consumes)
CHARTS = [
    ("make_stability_charts.py", "rank_stability"),
    ("make_seed_noise_charts.py", "seed_noise_by_group"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", type=Path,
                   default=Path(__file__).parents[2] / "results" / "esb_tuner_scaled",
                   help="results folder holding the shard CSVs "
                        "(default: results/esb_tuner_scaled)")
    p.add_argument("--shards", type=int, nargs="+", required=True,
                   help="shard numbers to analyze — always explicit, never defaulted")
    p.add_argument("--max-reps", type=int, default=None,
                   help="forwarded to rank_stability only: cap every shard at "
                        "its first N reps so unequal rep counts share a "
                        "comparable all-rep reference")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="directory for each test's --out CSV "
                        "(named <stem>.csv inside it); omit to only print")
    p.add_argument("--charts", action=argparse.BooleanOptionalAction, default=True,
                   help="render the chart PNGs from the saved CSVs after the "
                        "tests (needs --out-dir; --no-charts to skip)")
    args = p.parse_args()

    if not args.results_dir.is_dir():
        sys.exit(f"ERROR: results dir not found: {args.results_dir}\n"
                 "  Pass --results-dir explicitly if this isn't the core campaign.")
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        # Self-recording results: the exact rerun command lives next to the
        # outputs (and make_deck.py copies it into presenter notes).
        repo = Path(__file__).parents[2]
        try:
            rel = Path(__file__).resolve().relative_to(repo)
        except ValueError:
            rel = Path(__file__)
        cmd_line = "python " + " ".join(
            shlex.quote(a) for a in [str(rel).replace("\\", "/"), *sys.argv[1:]])
        (args.out_dir / "command.txt").write_text(
            f"# rerun-to-refresh — written by run_all_tests.py on "
            f"{datetime.now():%Y-%m-%d %H:%M}\n{cmd_line}\n", encoding="utf-8")

    here = Path(__file__).parent
    shards = [str(s) for s in args.shards]
    failed = []
    for test, extra, stem in TESTS:
        print(f"\n{'=' * 60}\nRUNNING {test} {' '.join(extra)}\n{'=' * 60}")
        cmd = [sys.executable, str(here / test), str(args.results_dir),
               "--shards", *shards, *extra]
        if args.max_reps and test == "rank_stability.py":
            cmd += ["--max-reps", str(args.max_reps)]
        if args.out_dir:
            cmd += ["--out", str(args.out_dir / f"{stem}.csv")]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            failed.append(f"{test} ({stem})")

    if args.charts and args.out_dir:
        for script, stem in CHARTS:
            csv = args.out_dir / f"{stem}.csv"
            if not csv.exists():
                failed.append(f"{script} (missing {csv.name})")
                continue
            print(f"\n{'=' * 60}\nRUNNING {script}\n{'=' * 60}")
            rc = subprocess.run([sys.executable, str(here / script), str(csv)]).returncode
            if rc != 0:
                failed.append(script)
    elif args.charts and not args.out_dir:
        print("\nNOTE: charts skipped — they need --out-dir for the CSVs.")

    if failed:
        sys.exit(f"\nFAILED: {', '.join(failed)}")
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
