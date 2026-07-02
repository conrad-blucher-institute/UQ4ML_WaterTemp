"""Sharding proofs (pure python — no TF, no numpy, no data files).

The load-bearing property: for ANY N, the union of shards 1..N is EXACTLY the
full job list — no gaps, no overlaps, order preserved. Plus: the shard string
parser fails loud on garbage, and artifact base names are unique across the
full Stage-C grid (regression lock for the dropout-collision fix).

Run:  python -m pytest esb/tests/test_sharding.py -q
  or: python esb/tests/test_sharding.py   (plain runner, no pytest needed)
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TUNE_DIR = _REPO_ROOT / "esb_03_2026_tune_experiment"
for _p in (str(_REPO_ROOT), str(_TUNE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from esb.sharding import (  # noqa: E402
    grid_fingerprint, job_key, parse_shard, shard_bounds, shard_slice, shard_suffix,
)


# ---------------------------------------------------------------------------
# parse_shard — loud on garbage, permissive on the documented forms
# ---------------------------------------------------------------------------
def test_parse_shard_valid():
    assert parse_shard("1/40") == (1, 40)
    assert parse_shard("40/40") == (40, 40)
    assert parse_shard(" 3 / 7 ") == (3, 7)
    assert parse_shard("1/1") == (1, 1)


def test_parse_shard_run_everything_forms():
    assert parse_shard(None) is None
    assert parse_shard("") is None
    assert parse_shard("  ") is None
    assert parse_shard("0/1") is None  # documented alias


def test_parse_shard_rejects_garbage():
    for bad in ("x", "1", "1/", "/40", "1/0", "0/2", "41/40", "-1/2", "1/40/2", "1.5/3"):
        try:
            parse_shard(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse_shard({bad!r}) should have raised ValueError")


# ---------------------------------------------------------------------------
# The partition property — the whole point of static sharding
# ---------------------------------------------------------------------------
def test_shards_tile_the_list_exactly():
    for m in (0, 1, 5, 119, 120, 4800):
        items = list(range(m))
        for n in (1, 2, 3, 7, 40, 200):
            rebuilt = []
            for k in range(1, n + 1):
                rebuilt.extend(shard_slice(items, (k, n)))
            assert rebuilt == items, f"gaps/overlap/reorder at M={m}, N={n}"


def test_shard_sizes_balanced():
    for m in (120, 4799, 4800):
        for n in (7, 40):
            sizes = [len(shard_slice(list(range(m)), (k, n))) for k in range(1, n + 1)]
            assert sum(sizes) == m
            assert max(sizes) - min(sizes) <= 1, f"unbalanced at M={m}, N={n}: {sizes}"


def test_shard_bounds_contiguous():
    m, n = 4800, 40
    prev_end = 0
    for k in range(1, n + 1):
        start, end = shard_bounds(m, k, n)
        assert start == prev_end, "blocks must be contiguous"
        prev_end = end
    assert prev_end == m


def test_none_means_everything():
    items = list(range(10))
    assert shard_slice(items, None) == items
    assert shard_suffix(None) == ""
    assert shard_suffix((3, 40)) == "_shard3of40"


# ---------------------------------------------------------------------------
# Fingerprint — same grid = same digest; edited grid = detectably different
# ---------------------------------------------------------------------------
def _mini_grid():
    from tuner_utils import GridSearchConfig
    return GridSearchConfig(
        model_type="MAPE", lead_times=[12, 48], cycles=[0, 1],
        activations=["relu", "selu"], num_layers_range=[1, 2],
        neurons_range=[16, 32], dropouts=[0.0, 0.1],
    )


def test_fingerprint_stable_and_sensitive():
    a = grid_fingerprint(_mini_grid().generate_configs())
    b = grid_fingerprint(_mini_grid().generate_configs())
    assert a == b, "same grid must fingerprint identically"
    edited = _mini_grid()
    edited.neurons_range = [16, 64]
    assert grid_fingerprint(edited.generate_configs()) != a, "grid edit must change fingerprint"


# ---------------------------------------------------------------------------
# Artifact-name uniqueness across the FULL Stage-C grid (dropout-fix lock)
# ---------------------------------------------------------------------------
def test_base_names_unique_across_full_grid():
    from tuner_utils import GridSearchConfig, job_base_name
    grid = GridSearchConfig(
        model_type="MAPE", lead_times=[12, 48, 96, 120], cycles=[0, 1, 2, 3],
        activations=["relu", "leaky_relu", "selu", "sigmoid", "tanh"],
        num_layers_range=[1, 2, 3], neurons_range=[16, 32, 64, 128, 256],
        dropouts=[0.0, 0.05, 0.1, 0.3],
    )
    configs = grid.generate_configs()
    assert len(configs) == 4800
    names = {job_base_name(c) for c in configs}
    assert len(names) == len(configs), (
        "artifact base names collide — two jobs would overwrite each other's "
        ".keras/_history/_scaler (the historical dropout bug)"
    )
    keys = {job_key(c) for c in configs}
    assert len(keys) == len(configs), "job_key must also be grid-unique"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    raise SystemExit(1 if failed else 0)
