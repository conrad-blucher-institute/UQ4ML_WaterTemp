"""Static multi-machine sharding — the SINGLE source of the shard math.

``--shard k/N`` (1-based: ``1/40`` = the first of 40 shards) splits the full
deterministic job list into N contiguous blocks; machine k runs only block k.
Contiguous blocks (not ``index % N``) so a shard is a human-readable range —
"machine 3 has jobs 241..360" — which makes manual hand-out and eyeballing a
progress CSV tractable (design doc §10).

Everything here is pure and import-light: the CLI uses it for ``--dry-run``
job listings, ``Config.validate()`` for loud early errors, ``base_tuner`` for
the actual slice, and the tests for the partition proof. One formula, four
consumers, zero drift.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Sequence

_SHARD_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def parse_shard(value: str | None) -> tuple[int, int] | None:
    """Parse ``'k/N'`` into ``(k, N)``; ``None``/empty/``'0/1'`` = run everything.

    k is 1-based (k in 1..N). ``'0/1'`` is accepted as an explicit
    run-everything alias. Raises ``ValueError`` (loud, actionable) otherwise.
    """
    if value is None or str(value).strip() == "":
        return None
    m = _SHARD_RE.match(str(value))
    if not m:
        raise ValueError(
            f"shard must look like 'k/N' (e.g. '1/40' = first of 40 shards), got {value!r}"
        )
    k, n = int(m.group(1)), int(m.group(2))
    if (k, n) == (0, 1):
        return None  # documented alias for "no sharding"
    if n < 1:
        raise ValueError(f"shard N must be >= 1, got {n} (from {value!r})")
    if not (1 <= k <= n):
        raise ValueError(
            f"shard k is 1-based and must be in 1..{n}, got k={k} (from {value!r})"
        )
    return (k, n)


def shard_bounds(n_jobs: int, k: int, n: int) -> tuple[int, int]:
    """[start, end) of shard k of N over ``n_jobs`` items (contiguous blocks).

    The standard balanced split: sizes differ by at most 1, and the blocks
    provably tile 0..n_jobs with no gaps and no overlaps.
    """
    return ((k - 1) * n_jobs) // n, (k * n_jobs) // n


def shard_slice(items: Sequence, shard: tuple[int, int] | None) -> Sequence:
    """The sub-list shard ``(k, N)`` owns; the untouched list when shard is None."""
    if shard is None:
        return items
    start, end = shard_bounds(len(items), *shard)
    return items[start:end]


def shard_suffix(shard: tuple[int, int] | None) -> str:
    """Filename suffix isolating one shard's mutable files (e.g. '_shard3of40').

    Empty when not sharding, so the no-flag layout is byte-identical to before.
    """
    return f"_shard{shard[0]}of{shard[1]}" if shard else ""


def job_key(config: dict[str, Any]) -> str:
    """Stable human-readable identity of one grid point (no run_num — the same
    config keeps its key across repetitions). Used for --dry-run listings and
    the campaign fingerprint."""
    return (
        f"{config['model_type']}_{config['lead_time']}h"
        f"_cycle{config['cycle']}_{config['activation']}"
        f"_{config['num_layers']}L_{config['neurons']}N"
        f"_d{config.get('dropout', 0.0)}"
    )


def grid_fingerprint(configs: Sequence[dict[str, Any]]) -> str:
    """Short digest of the full enumerated job list, recorded in provenance.

    Every shard of one campaign must carry the SAME fingerprint; if the grid
    is edited between handing out shard 2 and shard 3, the mismatch is
    detectable at merge time instead of silently producing a torn campaign.
    """
    joined = "\n".join(job_key(c) for c in configs)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
