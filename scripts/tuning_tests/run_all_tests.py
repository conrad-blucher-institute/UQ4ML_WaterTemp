"""Run every tuning test against the same shards in one command.

Thin dispatcher only: each test stays its own script and stays runnable
individually; this just invokes them in sequence with shared arguments.
metric_agreement runs three times — one per metric pair, including both
mae12 combos (the long-lead-time cold-tail divergence recheck).

Usage:
  python scripts/tuning_tests/run_all_tests.py results/esb_tuner_scaled --shards 1 11 21 23 28 31 --max-reps 10 --out-dir results/tuning_tests_phase1
"""

import argparse
import subprocess
import sys
from pathlib import Path

# (script, extra args, --out stem)
TESTS = [
    ("rank_stability.py", [], "rank_stability"),
    ("seed_noise_by_group.py", [], "seed_noise_by_group"),
    ("metric_agreement.py", ["--metrics", "val_mape", "val_mae"], "metric_agreement_mape_mae"),
    ("metric_agreement.py", ["--metrics", "val_mape", "val_mae12"], "metric_agreement_mape_mae12"),
    ("metric_agreement.py", ["--metrics", "val_mae", "val_mae12"], "metric_agreement_mae_mae12"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", help="results folder holding the shard CSVs")
    p.add_argument("--shards", type=int, nargs="+", required=True)
    p.add_argument("--max-reps", type=int, default=None,
                   help="forwarded to rank_stability only: cap every shard at "
                        "its first N reps so unequal rep counts share a "
                        "comparable all-rep reference")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="directory for each test's --out CSV "
                        "(named <stem>.csv inside it); omit to only print")
    args = p.parse_args()

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).parent
    shards = [str(s) for s in args.shards]
    failed = []
    for test, extra, stem in TESTS:
        print(f"\n{'=' * 60}\nRUNNING {test} {' '.join(extra)}\n{'=' * 60}")
        cmd = [sys.executable, str(here / test), args.folder, "--shards", *shards, *extra]
        if args.max_reps and test == "rank_stability.py":
            cmd += ["--max-reps", str(args.max_reps)]
        if args.out_dir:
            cmd += ["--out", str(args.out_dir / f"{stem}.csv")]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            failed.append(f"{test} ({stem})")
    if failed:
        sys.exit(f"\nFAILED: {', '.join(failed)}")
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
