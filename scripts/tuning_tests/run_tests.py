"""Run every tuning test against the same shards in one command.

Thin dispatcher only: each test stays its own script and stays runnable
individually; this just invokes them in sequence with shared arguments.

Usage:
  python scripts/tuning_tests/run_tests.py results/esb_tuner_scaled --shards 1 11 21 31
"""

import argparse
import subprocess
import sys
from pathlib import Path

TESTS = ["rank_stability.py", "seed_noise_by_group.py", "metric_agreement.py"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", help="results folder holding the shard CSVs")
    p.add_argument("--shards", type=int, nargs="+", required=True)
    args = p.parse_args()

    here = Path(__file__).parent
    shards = [str(s) for s in args.shards]
    failed = []
    for test in TESTS:
        print(f"\n{'=' * 60}\nRUNNING {test}\n{'=' * 60}")
        rc = subprocess.run([sys.executable, str(here / test),
                             args.folder, "--shards", *shards]).returncode
        if rc != 0:
            failed.append(test)
    if failed:
        sys.exit(f"\nFAILED: {', '.join(failed)}")
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
