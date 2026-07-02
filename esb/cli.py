"""esb command-line entry point — the only thing the user touches.

Subcommands:
  * ``esb run``      — launch an experiment (grid-search MAPE/MSE tuner).
  * ``esb profiles`` — list named profiles under esb/profiles/.

``run`` accepts, in increasing specificity:
  * bare flags with schema defaults filling the rest
        python -m esb run --loss mape --lead_times 12 --rotations 0
  * a named profile (resolved to esb/profiles/NAME.txt)
        python -m esb run --profile mape_smoke
  * an explicit @file of flags (one per line; whitespace-separated; # comments)
        python -m esb run @esb/profiles/mape.txt
  * ``--dry-run`` to print the fully-resolved config and exit (no training).

Profiles and @files share one mechanism: argparse ``fromfile_prefix_chars='@'``
with a whitespace/comment-aware line splitter. ``--profile NAME`` is rewritten to
``@<profiles_dir>/NAME.txt`` before parsing, so there is exactly one code path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from esb.config import Config, add_run_arguments, describe

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"

# Where does this run? (printed on `run`; mirrors the GUI blurb in the design doc)
_RUN_LOCATION_BANNER = (
    "Note: 'esb run' trains on THIS machine. Cluster/SLURM (Grendal) runs are "
    "manual by design — build a profile, copy it to the cluster, and submit there."
)


class _Parser(argparse.ArgumentParser):
    """argparse with whitespace/comment-aware @file expansion."""

    def convert_arg_line_to_args(self, line: str):
        line = line.split("#", 1)[0].strip()
        if not line:
            return []
        return line.split()


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="esb",
        description="ESB water-temperature experiment pipeline (single front door).",
        fromfile_prefix_chars="@",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run", help="Launch an experiment (grid-search MAPE/MSE tuner).",
        fromfile_prefix_chars="@", epilog=_RUN_LOCATION_BANNER,
    )
    run.add_argument("--profile", type=str, default=None,
                     help="Named profile under esb/profiles/ (e.g. mape_smoke).")
    run.add_argument("--dry-run", action="store_true",
                     help="Print the fully-resolved config and exit (no training).")
    add_run_arguments(run)

    sub.add_parser("profiles", help="List available named profiles.")
    return parser


def _resolve_profile_flag(argv: list[str]) -> list[str]:
    """Rewrite ``--profile NAME`` into ``@<profiles_dir>/NAME.txt`` (loud if missing)."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--profile":
            if i + 1 >= len(argv):
                _die("--profile requires a profile name.")
            name = argv[i + 1]
            path = PROFILES_DIR / f"{name}.txt"
            if not path.is_file():
                available = sorted(p.stem for p in PROFILES_DIR.glob("*.txt"))
                _die(f"profile {name!r} not found at {path}. Available: {available or '(none)'}")
            out.append(f"@{path}")
            i += 2
        else:
            out.append(tok)
            i += 1
    return out


def _expand_tokens(argv: list[str]) -> list[str]:
    """Expand any @file tokens into their flags (for explicit-flag detection)."""
    tokens: list[str] = []
    for tok in argv:
        if tok.startswith("@"):
            path = Path(tok[1:])
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.split("#", 1)[0].strip()
                    tokens.extend(line.split())
            else:
                tokens.append(tok)
        else:
            tokens.append(tok)
    return tokens


def _explicit_flags(expanded: list[str]) -> set[str]:
    """Field names whose --flag appears anywhere in the expanded token stream."""
    return {tok[2:] for tok in expanded if tok.startswith("--")}


def _die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"esb: error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    # UTF-8 stdout/stderr so em dashes / § / → print on Windows cp1252 consoles
    # without needing PYTHONIOENCODING=utf-8. errors="replace" = never crash.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    argv = list(sys.argv[1:] if argv is None else argv)

    # `profiles` needs no config parsing.
    if argv and argv[0] == "profiles":
        _list_profiles()
        return 0

    # Rewrite --profile to an @file BEFORE argparse sees it (one mechanism).
    rewritten = _resolve_profile_flag(argv)

    parser = _build_parser()
    ns = parser.parse_args(rewritten)

    if ns.command == "profiles":
        _list_profiles()
        return 0

    # Record which flags were explicitly passed (incl. via profile/@file) so
    # --debug only fills in where the user stayed silent.
    ns._explicit = _explicit_flags(_expand_tokens(rewritten))

    config = Config.from_namespace(ns)

    if ns.dry_run:
        ok = _print_resolved(config, validate=True)
        if ok:
            _print_shard_jobs(config)
        return 0

    try:
        config.validate()
    except Exception as e:  # ConfigError — print cleanly, no traceback
        print(f"esb: error: {e}", file=sys.stderr)
        return 2

    print(_RUN_LOCATION_BANNER + "\n")
    # Import here so config errors surface before the heavy TF import.
    from esb.pipeline import run_experiment
    run_experiment(config)
    return 0


def _list_profiles() -> None:
    profiles = sorted(PROFILES_DIR.glob("*.txt"))
    if not profiles:
        print(f"No profiles found in {PROFILES_DIR}")
        return
    print(f"Available profiles ({PROFILES_DIR}):")
    for p in profiles:
        first = ""
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("#"):
                first = s.lstrip("# ").strip()
                break
        print(f"  {p.stem:16s} {first}")


def _print_resolved(config: Config, validate: bool) -> bool:
    resolved = config.resolved()
    print("# Resolved configuration (--dry-run; no training performed)\n")
    print(json.dumps(resolved, indent=2, default=str))
    print()
    if validate:
        try:
            config.validate()
            print("# validation: OK")
        except Exception as e:
            print(f"# validation: FAILED\n{e}")
            return False
    return True


def _print_shard_jobs(config: Config) -> None:
    """With --shard, list exactly which jobs this shard would run (design §10).

    The listing comes from the SAME enumeration + slice the tuner uses
    (GridSearchConfig.generate_configs + esb.sharding.shard_bounds), so what is
    printed is what runs — no parallel implementation to drift.
    """
    from esb.sharding import job_key, parse_shard, shard_bounds

    shard = parse_shard(config.shard)  # validate() vetted the format already
    if shard is None:
        return
    from esb.pipeline import _build_grid_config  # light: no TF at module level

    jobs = _build_grid_config(config).generate_configs()
    start, end = shard_bounds(len(jobs), *shard)
    print(
        f"\n# shard {shard[0]}/{shard[1]}: jobs {start + 1}..{end} of {len(jobs)} "
        f"(per repetition; repetitions={config.repetitions})"
    )
    for c in jobs[start:end]:
        print(job_key(c))


if __name__ == "__main__":
    raise SystemExit(main())
