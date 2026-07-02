"""Orchestration — the ONE front-door entry into a run.

Stage B is a thin shell: ``run_experiment`` builds a ``GridSearchConfig`` and the
existing grid-search tuner from the validated ``Config``, then delegates the real
lead_times × rotations loop to the tuner's ``run_tuning`` (ProcessPoolExecutor +
ProgressTracker checkpoint/resume). No pipeline logic is reimplemented here.

The only run-level additions are the entry layer itself: best-effort seeding,
wiring the Config into the tuner via the backward-compatible hooks, and the
single unskippable provenance write (esb.io.results.finalize_run).
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

from esb.config import Config
from esb.io.results import finalize_run

# The grid-search tuner modules use bare imports (`from base_tuner import ...`),
# so both the repo root and the tune-experiment dir must be importable. These
# inserts also propagate to spawned worker processes (multiprocessing copies the
# parent's sys.path), matching how run_all_tuners.py is launched.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TUNE_DIR = _REPO_ROOT / "esb_03_2026_tune_experiment"
for _p in (_TUNE_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _apply_seed(seed: int) -> None:
    """Best-effort determinism. Note: full bitwise weight reproducibility under
    TensorFlow + multiprocessing is NOT guaranteed; the deterministic invariant
    we lock for the golden baseline is the model.fit-INPUT hash (data prep has no
    randomness), not trained weights."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def _build_grid_config(config: Config):
    """Map the Config grid fields onto GridSearchConfig (rotation -> cycle, R6)."""
    from tuner_utils import GridSearchConfig
    return GridSearchConfig(
        model_type=config.loss.upper(),
        lead_times=list(config.lead_times),
        cycles=list(config.rotations),          # esb 'rotation' == tuner 'cycle'
        activations=list(config.activations),
        num_layers_range=list(config.num_layers),
        neurons_range=list(config.neurons),
        dropouts=list(config.dropout),          # Stage C search-space axis (§9)
    )


def _build_config_overrides(config: Config) -> dict:
    """Per-config keys injected into every grid entry (consumed by _train_model)."""
    overrides = {
        "path_to_data": config.data_path,
        "input_structure": config.input_structure,
        "independent_year": config.independent_year,
        # Scale stage (C1): opt-in, off by default. _train_model routes arrays
        # through esb.stages.scale only when this is True.
        "scale": config.scale,
        "atp_hours_back": config.atp_hours_back,
        "wtp_hours_back": config.wtp_hours_back,
        "pred_atp_interval": config.pred_atp_interval,
        "IPPOffset": config.ipp_offset,
        "output_units": config.output_units,
        "output_activation": config.output_activation,
        "learning_rate": config.learning_rate,
        # R5 callback knobs (defaults equal the old literals -> byte-identical)
        "early_stop_patience": config.early_stop_patience,
        "lr_reducer_patience": config.lr_reducer_patience,
        "min_delta": config.min_delta,
        "call_back_monitor": config.call_back_monitor,
    }
    # Only pin batch_size when explicitly set; None -> tuner infers full-train-set.
    if config.batch_size is not None:
        overrides["batch_size"] = config.batch_size
    return overrides


def _tuner_class(loss: str):
    if loss == "mape":
        from mape_tuner import MAPETuner
        return MAPETuner
    if loss == "mse":
        from mse_tuner import MSETuner
        return MSETuner
    raise ValueError(f"unsupported loss {loss!r} (validation should have caught this)")


def run_experiment(config: Config) -> None:
    """Launch the grid-search tuner for the resolved Config (single front door)."""
    config.validate()  # defensive: never train on an invalid config
    _apply_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keras_save_dir = output_dir / "keras_files"

    # preparingData()'s verbose debug logs land in the run folder (matches
    # run_all_tuners.py behavior).
    os.environ["TUNER_DEBUG_LOG_DIR"] = str(output_dir)
    # Avoid BLAS/OpenMP oversubscription with multiprocessing (as run_all_tuners).
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    grid_config = _build_grid_config(config)
    config_overrides = _build_config_overrides(config)

    # Sharding (design §10): parse once (validate() already vetted the format).
    # Provenance records the full-grid fingerprint + this shard's block so every
    # shard of one campaign is traceable, and a grid edited between hand-outs
    # (fingerprint mismatch) is caught at merge time instead of silently
    # producing a torn campaign.
    from esb.sharding import grid_fingerprint, parse_shard, shard_bounds, shard_suffix
    shard = parse_shard(config.shard)
    all_jobs = grid_config.generate_configs()
    resolved = config.resolved()
    resolved["_jobs"] = {
        "grid_fingerprint": grid_fingerprint(all_jobs),
        "jobs_total": len(all_jobs),
    }
    if shard is not None:
        start, end = shard_bounds(len(all_jobs), *shard)
        resolved["_jobs"]["shard"] = {
            "k": shard[0], "N": shard[1],
            "jobs_start": start + 1, "jobs_end": end,  # 1-based inclusive range
            "jobs_in_shard": end - start,
        }

    # Provenance FIRST: a run can never exist without its record, even if it
    # later crashes. This is the single, unskippable write site. Per-shard
    # filename so machines never collide when result trees are union-merged.
    prov_path = finalize_run(
        output_dir, resolved, filename=f"run_provenance{shard_suffix(shard)}.json"
    )
    print(f"[esb] wrote provenance: {prov_path}")

    TunerClass = _tuner_class(config.loss)
    tuner = TunerClass(
        output_dir,
        max_workers=config.workers,
        keras_save_dir=keras_save_dir,
        verbose=config.verbose,
        shard=shard,
    )
    # Drive the tuner from the Config single-source-of-truth via the additive hooks.
    tuner.grid_config = grid_config
    tuner.config_overrides = config_overrides
    tuner.default_epochs = config.epochs

    n_configs = grid_config.count_configs()
    print(
        f"[esb] loss={config.loss} | grid={n_configs} configs "
        f"(lead_times={config.lead_times} × rotations={config.rotations} × "
        f"activations={len(config.activations)} × layers={config.num_layers} × "
        f"neurons={config.neurons}) | epochs={config.epochs} | workers={config.workers} "
        f"| repetitions={config.repetitions} | output={output_dir}"
    )
    if shard is not None:
        info = resolved["_jobs"]["shard"]
        print(
            f"[esb] shard {info['k']}/{info['N']}: jobs {info['jobs_start']}..{info['jobs_end']} "
            f"of {resolved['_jobs']['jobs_total']} (per repetition) | "
            f"fingerprint {resolved['_jobs']['grid_fingerprint']}"
        )

    for run_idx in range(int(config.repetitions)):
        run_id = run_idx + 1
        print(f"\n[esb] === {config.loss.upper()} run {run_id}/{config.repetitions} ===")
        tuner.run_tuning(run_num=run_id, max_configs=config.max_models)

    print(f"\n[esb] done. Results + provenance in {output_dir}")
