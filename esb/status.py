"""`esb status` — read-only campaign progress view (per-shard + artifact check).

Usage:  python -m esb status --profile mape_scaled
        (same flags as `esb run`; the profile tells status which grid/output_dir
         the campaign uses — status itself never trains or writes anything)

For every ``*_progress_shard{k}of{N}.csv`` in output_dir (plus the legacy
unsharded ``*_progress.csv``) it reports completed/expected job counts and a
throughput-based ETA, and cross-checks the .keras artifacts:

  * ROW-BUT-NO-FILE — a job recorded completed whose .keras is missing
    (lost artifact: the model cannot be recovered without retraining).
  * FILE-BUT-NO-ROW — a .keras present with no completed row (orphan:
    resume will RETRAIN and overwrite it).

Guardrail: expected job sets come from the SAME enumeration + slice the tuner
uses (GridSearchConfig.generate_configs + esb.sharding.shard_bounds) and the
SAME identity/name functions (ProgressTracker._make_key, job_base_name) — no
parallel math, so status can never disagree with the tuner.
"""
from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

from esb.config import Config

_SHARD_CSV = re.compile(r"^(?P<prefix>\w+)_progress_shard(?P<k>\d+)of(?P<n>\d+)\.csv$")
_LEGACY_CSV = re.compile(r"^(?P<prefix>\w+)_progress\.csv$")


def _expected_jobs(config: Config, shard: tuple[int, int] | None) -> list[dict]:
    """The tuner's own enumeration + slice, with run_num attached per repetition."""
    from esb.pipeline import _build_grid_config
    from esb.sharding import shard_slice

    jobs = _build_grid_config(config).generate_configs()
    jobs = shard_slice(jobs, shard)
    return [
        {**job, "run_num": rep}
        for rep in range(1, int(config.repetitions) + 1)
        for job in jobs
    ]


def _int(x, default: int = 0) -> int:
    """Tolerant int for CSV cells (malformed/empty -> default, never a crash)."""
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _fmt_eta(rows: list[dict], done: int, remaining: int, workers: int) -> str:
    """ETA = remaining x median(duration_sec) / workers.

    Per-job durations are measured inside the worker, so idle gaps between a
    shard's turns in the sequential campaign loop cannot inflate the estimate
    (the old timestamp-span method showed 30-90 h for a ~2 h shard). Median is
    robust to stragglers; no sigma-based cut needed. Falls back to the span
    method for CSVs that predate the duration_sec column.
    """
    if remaining == 0:
        return "complete"
    durations = []
    for r in rows:
        try:
            durations.append(float(r.get("duration_sec") or ""))
        except ValueError:
            pass
    if durations:
        durations.sort()
        n = len(durations)
        median = (durations[n // 2] if n % 2 else
                  (durations[n // 2 - 1] + durations[n // 2]) / 2)
        hours = remaining * median / max(workers, 1) / 3600
        return (f"~{hours:.1f} h compute at {workers} worker(s) "
                f"(median {median:.0f}s/job; pass --workers to match your run)")
    # fallback: timestamp span (pre-duration_sec CSVs) — inflated by idle gaps
    stamps = sorted(r["timestamp"] for r in rows if r.get("timestamp"))
    if len(stamps) < 2 or done < 2:
        return "n/a"
    try:
        span = (datetime.fromisoformat(stamps[-1]) - datetime.fromisoformat(stamps[0])).total_seconds()
    except ValueError:
        return "n/a"
    if span <= 0:
        return "n/a"
    rate = (done - 1) / span  # jobs/sec between first and last row
    hours = remaining / rate / 3600
    return f"~{hours:.1f} h at this shard's observed rate (span-based; may include idle gaps)"


def run_status(config: Config) -> int:
    """Print the campaign status; exit code 0 = clean, 1 = anomalies, 2 = unusable."""
    from esb.sharding import parse_shard
    import sys

    # tuner_utils lives in the tune-experiment dir (same sys.path trick as pipeline).
    _tune_dir = Path(__file__).resolve().parent.parent / "esb_03_2026_tune_experiment"
    if str(_tune_dir) not in sys.path:
        sys.path.insert(0, str(_tune_dir))
    from tuner_utils import ProgressTracker, job_base_name

    output_dir = Path(config.output_dir)
    keras_dir = output_dir / "keras_files"
    if not output_dir.is_dir():
        print(f"esb status: output_dir does not exist: {output_dir}")
        return 2

    # -- discover progress CSVs ------------------------------------------------
    sharded: dict[int, tuple[int, Path]] = {}   # k -> (N, path)
    legacy: list[Path] = []
    for p in sorted(output_dir.glob("*_progress*.csv")):
        m = _SHARD_CSV.match(p.name)
        if m:
            k, n = int(m["k"]), int(m["n"])
            sharded[k] = (n, p)
        elif _LEGACY_CSV.match(p.name):
            legacy.append(p)

    ns = {n for n, _ in sharded.values()}
    if len(ns) > 1:
        print(f"esb status: MIXED shard denominators found in {output_dir}: N={sorted(ns)}. "
              f"These are different partitions of the grid — results are ambiguous. "
              f"Keep exactly one N per output_dir.")
        return 2
    N = ns.pop() if ns else None

    anomalies = 0
    total_done = total_expected = 0
    blocks: list[tuple[int, list[str]]] = []  # (shard k or 0, that block's lines)

    def check_block(label: str, shard: tuple[int, int] | None, rows: list[dict]) -> None:
        nonlocal anomalies, total_done, total_expected
        lines: list[str] = []
        expected = _expected_jobs(config, shard)
        expected_keys = {ProgressTracker._make_key(j) for j in expected}
        done_rows = [r for r in rows if r.get("status") == "completed"]
        done_keys = {ProgressTracker._make_key(r) for r in done_rows}

        known_done = done_keys & expected_keys
        stray = done_keys - expected_keys  # rows not in this grid (edited grid? other reps)
        missing = len(expected_keys) - len(known_done)
        total_done += len(known_done)
        total_expected += len(expected_keys)

        # artifact cross-check: completed row -> .keras must exist
        by_key = {ProgressTracker._make_key(j): j for j in expected}
        lost = [k for k in sorted(known_done)
                if not (keras_dir / f"{job_base_name(by_key[k])}.keras").is_file()]

        pct = 100.0 * len(known_done) / len(expected_keys) if expected_keys else 0.0
        state = ("complete" if missing == 0 else
                 "in progress" if known_done else "not started")
        lines.append(f"  {label:18s} {len(known_done):5d}/{len(expected_keys):<5d} "
                     f"({pct:5.1f}%)  {state:12s} "
                     f"{_fmt_eta(done_rows, len(known_done), missing, int(config.workers))}")
        if stray:
            # Distinguish "later repetitions" (informational — rerun status with a
            # higher --repetitions to audit them) from truly-unknown rows (loud).
            reps = {_int(r.get("run_num")) for r in done_rows
                    if ProgressTracker._make_key(r) in stray}
            extra_reps = sorted(x for x in reps if x > int(config.repetitions))
            n_extra = sum(1 for r in done_rows
                          if ProgressTracker._make_key(r) in stray
                          and _int(r.get("run_num")) > int(config.repetitions))
            if n_extra:
                lines.append(f"    -- {n_extra} row(s) from later repetition(s) "
                             f"{extra_reps} — rerun with --repetitions "
                             f"{max(extra_reps)} to include them in the counts")
            unknown = len(stray) - n_extra
            if unknown:
                anomalies += unknown
                lines.append(f"    !! {unknown} completed row(s) NOT in the current grid "
                             f"at any repetition (grid edited since hand-out?)")
        if lost:
            anomalies += len(lost)
            lines.append(f"    !! ROW-BUT-NO-FILE: {len(lost)} completed job(s) missing their "
                         f".keras in {keras_dir.name}/ — e.g. {lost[0]}")

        # file-but-no-row within this shard: crashed between model.save and the
        # CSV append — resume will RETRAIN and overwrite (harmless, but visible).
        pending = [k for k in sorted(expected_keys - known_done)
                   if (keras_dir / f"{job_base_name(by_key[k])}.keras").is_file()]
        if pending:
            lines.append(f"    -- FILE-BUT-NO-ROW: {len(pending)} .keras exist without a "
                         f"completed row (resume will retrain them) — e.g. {pending[0]}")
        blocks.append((shard[0] if shard else 0, lines))

    print(f"esb status — {output_dir}")
    print(f"grid: loss={config.loss} | repetitions={config.repetitions}"
          + (f" | shards of N={N}" if N else " | unsharded"))
    if legacy:
        for p in legacy:
            check_block(p.name, None, _read_rows(p))
    for k in sorted(sharded):
        n, p = sharded[k]
        check_block(f"shard {k}/{n}", parse_shard(f"{k}/{n}"), _read_rows(p))
    if N:
        for k in range(1, N + 1):
            if k not in sharded:
                expected = _expected_jobs(config, (k, N))
                total_expected += len(expected)
                blocks.append((k, [f"  {'shard %d/%d' % (k, N):18s} {0:5d}/{len(expected):<5d} "
                                   f"(  0.0%)  not started   (no progress CSV)"]))
    blocks.sort(key=lambda b: b[0])
    print("\n".join(line for _, block in blocks for line in block))

    # -- global orphan check: .keras files no expected job would produce -------
    # Judged against ALL repetitions seen anywhere (not just --repetitions), so
    # later-rep artifacts are not misflagged as orphans.
    if keras_dir.is_dir():
        max_run = max(
            [int(config.repetitions)]
            + [_int(r.get("run_num")) for _, p in sharded.values()
               for r in _read_rows(p)]
            + [_int(r.get("run_num")) for p in legacy for r in _read_rows(p)]
        )
        from esb.pipeline import _build_grid_config
        base_jobs = _build_grid_config(config).generate_configs()
        all_expected_names = {
            f"{job_base_name({**j, 'run_num': rep})}.keras"
            for rep in range(1, max_run + 1)
            for j in base_jobs
        }
        orphans = sorted(f.name for f in keras_dir.glob("*.keras")
                         if f.name not in all_expected_names)
        if orphans:
            anomalies += len(orphans)
            print(f"  !! FILE-BUT-NO-EXPECTED-JOB: {len(orphans)} .keras file(s) no job in this "
                  f"grid would write (old grid/reps?) — e.g. {orphans[0]}")

    pct = 100.0 * total_done / total_expected if total_expected else 0.0
    print(f"\nTOTAL: {total_done}/{total_expected} jobs ({pct:.1f}%)"
          + (f" — {anomalies} anomal{'y' if anomalies == 1 else 'ies'} (see !! above)"
             if anomalies else " — no anomalies"))
    return 1 if anomalies else 0
