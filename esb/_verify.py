"""Stage B golden-baseline tooling (forward-correctness, not reproduce-the-past).

There is no runnable legacy entry to diff against (R1: the old pipeline crashed),
so the baseline is the FIRST working run through the new entry. The invariant we
lock is the **model.fit-INPUT hash**: the x/y arrays + shapes + the hyperparams
that reach ``model.fit`` / ``model.compile``. Data prep has no randomness (sorted
glob, fixed CSVs, vectorized shift, deterministic split), so this hash is
deterministic by construction — exactly the contract Stages C and D must
reproduce byte-for-byte when they refactor data prep / add scaling.

This reproduces what ``mape_tuner._train_model`` feeds the model (it uses
``preparingData(...)[:4]`` and ``batch_size = len(x_train)`` when unset), WITHOUT
instrumenting the tuner.

Usage (from repo root):
  python -m esb._verify freeze   # compute hash, write ./_golden_baseline/
  python -m esb._verify check    # recompute, assert identical to frozen baseline
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from esb.cli import _build_parser, _resolve_profile_flag
from esb.config import Config

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BASELINE_DIR = _REPO_ROOT / "_golden_baseline"
sys.path.insert(0, str(_REPO_ROOT))


def _smoke_config() -> Config:
    """Resolve the mape_smoke profile exactly as the CLI would."""
    argv = _resolve_profile_flag(["run", "--profile", "mape_smoke"])
    ns = _build_parser().parse_args(argv)
    return Config.from_namespace(ns).validate()


def _arr_digest(arr) -> dict:
    import numpy as np
    a = np.ascontiguousarray(arr)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
    }


def compute_fit_inputs(config: Config) -> dict:
    """Reproduce the single-config model.fit inputs + compiled hyperparams."""
    from src.helper.utils_mse_crps import preparingData
    import numpy as np

    lead_time = int(config.lead_times[0])
    rotation = int(config.rotations[0])

    data = preparingData(
        path_to_data=config.data_path,
        input_structure=config.input_structure,
        independent_year=config.independent_year,
        input_hours_forecast=lead_time,
        atp_hours_back=config.atp_hours_back,
        wtp_hours_back=config.wtp_hours_back,
        pred_atp_interval=config.pred_atp_interval,
        IPPOffset=config.ipp_offset,
        cycle=rotation,
        model=config.loss.upper(),
        verbose=0,
    )
    X_train, y_train, X_val, y_val = data[:4]
    x_train = np.asarray(getattr(X_train, "values", X_train), dtype=float)
    y_train = np.asarray(getattr(y_train, "values", y_train), dtype=float)
    x_val = np.asarray(getattr(X_val, "values", X_val), dtype=float)
    y_val = np.asarray(getattr(y_val, "values", y_val), dtype=float)

    batch_size = config.batch_size if config.batch_size is not None else int(x_train.shape[0])

    payload = {
        "config": {
            "loss": config.loss, "lead_time": lead_time, "rotation": rotation,
            "input_structure": config.input_structure,
            "atp_hours_back": config.atp_hours_back,
            "wtp_hours_back": config.wtp_hours_back,
            "pred_atp_interval": config.pred_atp_interval,
            "activation": config.activations[0],
            "num_layers": config.num_layers[0], "neurons": config.neurons[0],
            "output_units": config.output_units,
            "output_activation": config.output_activation,
            "learning_rate": config.learning_rate,
        },
        "fit_inputs": {
            "x_train": _arr_digest(x_train), "y_train": _arr_digest(y_train),
            "x_val": _arr_digest(x_val), "y_val": _arr_digest(y_val),
        },
        "fit_hyperparams": {
            "epochs": config.epochs, "batch_size": batch_size,
            "compiled_loss": "mape", "compiled_metrics": ["mape", "mae"],
        },
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return payload


def freeze() -> int:
    config = _smoke_config()
    payload = compute_fit_inputs(config)
    _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    (_BASELINE_DIR / "fit_inputs.json").write_text(json.dumps(payload, indent=2, default=str))
    (_BASELINE_DIR / "resolved_config.json").write_text(
        json.dumps(config.resolved(), indent=2, default=str)
    )
    # Copy the run-level artifacts produced by the actual run, if present.
    run_dir = Path(config.output_dir)
    for name in ("run_provenance.json", "mape_progress.csv"):
        src = run_dir / name
        if src.is_file():
            (_BASELINE_DIR / name).write_text(src.read_text())
    print(f"[freeze] fit-input digest: {payload['digest']}")
    print(f"[freeze] wrote baseline -> {_BASELINE_DIR}")
    return 0


def check() -> int:
    frozen_path = _BASELINE_DIR / "fit_inputs.json"
    if not frozen_path.is_file():
        print(f"[check] no frozen baseline at {frozen_path}; run `freeze` first.")
        return 1
    frozen = json.loads(frozen_path.read_text())
    current = compute_fit_inputs(_smoke_config())
    if current["digest"] == frozen["digest"]:
        print(f"[check] PASS — fit-input digest identical: {current['digest']}")
        return 0
    print("[check] FAIL — fit-input digest differs!")
    print(f"  frozen : {frozen['digest']}")
    print(f"  current: {current['digest']}")
    for k in ("x_train", "y_train", "x_val", "y_val"):
        f, c = frozen["fit_inputs"][k], current["fit_inputs"][k]
        if f != c:
            print(f"  {k}: frozen={f} current={c}")
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "check"
    if cmd == "freeze":
        return freeze()
    if cmd == "check":
        return check()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
