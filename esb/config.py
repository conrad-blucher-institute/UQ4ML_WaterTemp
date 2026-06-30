"""Config schema — the SINGLE SOURCE OF TRUTH for the esb pipeline.

Every option is declared exactly once in ``FIELDS`` as a typed ``Field`` with
metadata (type, default, choices, help, stage). From that one declaration we:

  * generate the argparse CLI (:func:`build_parser`), and
  * stay introspectable so a future GUI can render a form from the same schema
    (:func:`describe`).

Add an option here and it appears as a CLI flag *and* a GUI field automatically —
the two can never drift (see ESB_PIPELINE_DESIGN.md §8).

Guardrail support:
  * Hidden ≠ invisible — every default lives here, inspectable via ``--dry-run``.
  * Magic needs loud errors — :meth:`Config.validate` collects *all* problems and
    raises a single ``ConfigError`` at startup, before any TensorFlow import or
    training begins.

Stage B scope: the grid-search MAPE tuner (loss=mape; mse also available). The
CRPS ensemble path and the RandomSearch tuner are parked (see Stage A report).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repo root = parent of the esb/ package directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Allowed value sets (one definition each; argparse + validate() both use these)
# ---------------------------------------------------------------------------
ALLOWED_LEAD_TIMES = [12, 24, 48, 72, 96, 108, 120]
ALLOWED_ROTATIONS = [0, 1, 2, 3]
ALLOWED_STRUCTURES = ["descending", "ascending"]
ALLOWED_INDEPENDENT = ["cycle", "2021"]
ALLOWED_ACTIVATIONS = ["relu", "leaky_relu", "selu", "sigmoid", "tanh"]
ALLOWED_LOSS = ["mape", "mse"]          # tuners that exist and run today
PARKED_LOSS = ["crps"]                  # ensemble path — deferred (loud error)

# Default data location (validated to exist at startup).
DEFAULT_DATA_PATH = str(_REPO_ROOT / "data" / "ESB_datasets")


class ConfigError(ValueError):
    """Raised (once, with all problems) when a Config fails validation."""


@dataclass(frozen=True)
class Field:
    """One configurable option — declared once, consumed everywhere."""

    name: str
    type: type            # scalar python type, or the ITEM type for lists
    default: Any
    help: str
    stage: str            # which pipeline stage consumes it (doc/introspection)
    choices: tuple | None = None
    is_list: bool = False
    is_flag: bool = False  # store_true boolean
    optional: bool = False  # may be None (e.g. batch_size, max_models)

    @property
    def flag(self) -> str:
        return f"--{self.name}"


# ---------------------------------------------------------------------------
# THE SCHEMA. Order here = order in --help and in --dry-run output.
# ---------------------------------------------------------------------------
FIELDS: list[Field] = [
    # --- read --------------------------------------------------------------
    Field("data_path", str, DEFAULT_DATA_PATH,
          "Directory of ESB year CSVs (sorted; first is held out as the independent year).",
          stage="read"),
    Field("independent_year", str, "cycle",
          "Which year is the held-out test set: 'cycle' (rotation) or '2021'.",
          stage="read", choices=tuple(ALLOWED_INDEPENDENT)),
    # --- features ----------------------------------------------------------
    Field("input_structure", str, "descending",
          "Feature column ordering; drives the col_start invariant (1 desc / 3 asc).",
          stage="features", choices=tuple(ALLOWED_STRUCTURES)),
    Field("atp_hours_back", int, 24, "Air-temperature lag hours used as features.",
          stage="features"),
    Field("wtp_hours_back", int, 24, "Water-temperature lag hours used as features.",
          stage="features"),
    Field("pred_atp_interval", int, 1, "Forecast air-temperature sampling interval.",
          stage="features"),
    Field("ipp_offset", float, 0.0, "IPP perturbation offset added to inputs.",
          stage="features"),
    # --- grid (search space) ----------------------------------------------
    Field("lead_times", int, [12, 48, 96, 120],
          "Forecast lead times (hours) to search over.",
          stage="grid", choices=tuple(ALLOWED_LEAD_TIMES), is_list=True),
    Field("rotations", int, [0, 1, 2, 3],
          "Temporal rotations to search over (a.k.a. cycle; 4-year rotation).",
          stage="grid", choices=tuple(ALLOWED_ROTATIONS), is_list=True),
    Field("activations", str, ["relu", "leaky_relu", "selu", "sigmoid", "tanh"],
          "Hidden-layer activation functions to search over.",
          stage="grid", choices=tuple(ALLOWED_ACTIVATIONS), is_list=True),
    Field("num_layers", int, [1, 2, 3], "Hidden-layer counts to search over.",
          stage="grid", is_list=True),
    Field("neurons", int, [16, 32, 64, 100, 128, 256],
          "Neurons-per-layer counts to search over.", stage="grid", is_list=True),
    # --- scale -------------------------------------------------------------
    Field("scale", bool, False,
          "Standardize inputs: fit a StandardScaler on TRAINING data only, "
          "transform val/test, persist the scaler (.joblib) and reuse it for "
          "inference. OFF by default (behavior-preserving; the unscaled golden "
          "baseline still holds).",
          stage="scale", is_flag=True),
    # --- model -------------------------------------------------------------
    Field("loss", str, "mape",
          "Loss / tuner to launch: 'mape' or 'mse' (crps parked).",
          stage="model", choices=tuple(ALLOWED_LOSS + PARKED_LOSS)),
    Field("output_units", int, 1,
          "Output-layer units (CRPS ensemble needs >1; mape/mse use 1).",
          stage="model"),
    Field("output_activation", str, "linear", "Output-layer activation.", stage="model"),
    Field("learning_rate", float, 0.001, "Adam learning rate.", stage="train"),
    # --- train -------------------------------------------------------------
    Field("epochs", int, 200000, "Max training epochs (early stopping may cut short).",
          stage="train"),
    Field("batch_size", int, None,
          "Batch size; None = full training set (the MSE/MAPE default).",
          stage="train", optional=True),
    Field("early_stop_patience", int, 25, "EarlyStopping patience (epochs).", stage="train"),
    Field("lr_reducer_patience", int, 15, "ReduceLROnPlateau patience (epochs).", stage="train"),
    Field("min_delta", float, 0.001, "Min improvement for early stop / LR reduce.",
          stage="train"),
    Field("call_back_monitor", str, "val_loss", "Quantity callbacks monitor.", stage="train"),
    # --- run / orchestration ----------------------------------------------
    Field("output_dir", str, str(_REPO_ROOT / "results" / "esb_tuner"),
          "Run output directory (provenance + progress CSV land here).", stage="run"),
    Field("workers", int, 4, "Parallel worker processes.", stage="run"),
    Field("repetitions", int, 1, "Independent repeated runs (each gets a run_num).",
          stage="run"),
    Field("max_models", int, None, "Cap configs explored (debug subset); None = all.",
          stage="run", optional=True),
    Field("seed", int, 42, "Seed applied (best-effort) before training.", stage="run"),
    Field("verbose", int, 0, "Verbosity 0-3.", stage="run"),
    Field("debug", bool, False,
          "Debug mode: 2 epochs / 10 models / 1 worker / 2 reps (overridable).",
          stage="run", is_flag=True),
]

FIELDS_BY_NAME: dict[str, Field] = {f.name: f for f in FIELDS}


# ---------------------------------------------------------------------------
# argparse generation
# ---------------------------------------------------------------------------
def add_run_arguments(parser) -> None:
    """Add every schema field to ``parser`` as a flag (generated from FIELDS)."""
    for f in FIELDS:
        if f.is_flag:
            parser.add_argument(f.flag, action="store_true", default=False, help=f.help)
        elif f.is_list:
            parser.add_argument(
                f.flag, nargs="+", type=f.type,
                choices=list(f.choices) if f.choices else None,
                default=list(f.default), metavar=f.name.upper(),
                help=f.help + f" (default: {f.default})",
            )
        else:
            parser.add_argument(
                f.flag, type=f.type,
                choices=list(f.choices) if f.choices else None,
                default=f.default,
                help=f.help + f" (default: {f.default})",
            )


# ---------------------------------------------------------------------------
# Resolved config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """A fully-resolved, validated configuration for one experiment run."""

    values: dict[str, Any]

    @classmethod
    def from_namespace(cls, ns) -> "Config":
        """Build from an argparse namespace, applying --debug conveniences."""
        values = {f.name: getattr(ns, f.name) for f in FIELDS}

        # --debug applies conveniences ONLY where the user left the default,
        # so explicit flags always win (hidden ≠ invisible). The CLI records the
        # set of explicitly-passed flag names on ns._explicit.
        if values.get("debug"):
            explicit = getattr(ns, "_explicit", set())
            debug_defaults = {"epochs": 2, "max_models": 10, "workers": 1, "repetitions": 2}
            for k, dv in debug_defaults.items():
                if k not in explicit:
                    values[k] = dv

        return cls(values)

    # -- access -------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as e:  # pragma: no cover - defensive
            raise AttributeError(name) from e

    def resolved(self) -> dict[str, Any]:
        """Full resolved config (for --dry-run and provenance)."""
        out = dict(self.values)
        out["_derived"] = {
            "keras_save_dir": str(Path(self.values["output_dir"]) / "keras_files"),
            "repo_root": str(_REPO_ROOT),
            "allowed_lead_times": ALLOWED_LEAD_TIMES,
        }
        return out

    # -- validation ---------------------------------------------------------
    def validate(self) -> "Config":
        """Fail loud and early: collect ALL problems, raise once."""
        errs: list[str] = []
        v = self.values

        if v["loss"] in PARKED_LOSS:
            errs.append(
                f"loss={v['loss']!r} is parked (the CRPS ensemble / RandomSearch path "
                f"is a future feature). Use one of {ALLOWED_LOSS}."
            )
        elif v["loss"] not in ALLOWED_LOSS:
            errs.append(f"loss={v['loss']!r} not in {ALLOWED_LOSS}.")

        # data path must exist and hold enough CSVs (4 training + 1 independent).
        data_path = Path(v["data_path"])
        if not data_path.is_dir():
            errs.append(f"data_path does not exist or is not a directory: {data_path}")
        else:
            csvs = sorted(data_path.glob("*.csv"))
            if len(csvs) < 2:
                errs.append(
                    f"data_path {data_path} has {len(csvs)} CSV(s); need at least 2 "
                    f"(>=1 independent year + >=1 training year)."
                )

        for lt in v["lead_times"]:
            if lt not in ALLOWED_LEAD_TIMES:
                errs.append(f"lead_time {lt} not in {ALLOWED_LEAD_TIMES}.")
        for r in v["rotations"]:
            if r not in ALLOWED_ROTATIONS:
                errs.append(f"rotation {r} not in {ALLOWED_ROTATIONS}.")
        for a in v["activations"]:
            if a not in ALLOWED_ACTIVATIONS:
                errs.append(f"activation {a!r} not in {ALLOWED_ACTIVATIONS}.")
        for n in v["num_layers"]:
            if n < 1:
                errs.append(f"num_layers entry {n} must be >= 1.")
        for n in v["neurons"]:
            if n < 1:
                errs.append(f"neurons entry {n} must be >= 1.")

        if v["input_structure"] not in ALLOWED_STRUCTURES:
            errs.append(f"input_structure {v['input_structure']!r} not in {ALLOWED_STRUCTURES}.")
        if v["independent_year"] not in ALLOWED_INDEPENDENT:
            errs.append(f"independent_year {v['independent_year']!r} not in {ALLOWED_INDEPENDENT}.")

        # output-unit consistency (the rule, even though crps is parked).
        if v["loss"] in ("mape", "mse") and v["output_units"] != 1:
            errs.append(f"loss={v['loss']} expects output_units=1, got {v['output_units']}.")
        if v["output_units"] < 1:
            errs.append(f"output_units must be >= 1, got {v['output_units']}.")

        for name in ("epochs", "workers", "repetitions"):
            if v[name] < 1:
                errs.append(f"{name} must be >= 1, got {v[name]}.")
        if v["batch_size"] is not None and v["batch_size"] < 1:
            errs.append(f"batch_size must be >= 1 or None, got {v['batch_size']}.")
        if v["max_models"] is not None and v["max_models"] < 1:
            errs.append(f"max_models must be >= 1 or None, got {v['max_models']}.")

        if not v["lead_times"]:
            errs.append("lead_times is empty — nothing to search.")
        if not v["rotations"]:
            errs.append("rotations is empty — nothing to search.")
        if not v["activations"]:
            errs.append("activations is empty — nothing to search.")

        if errs:
            bullet = "\n  - ".join(errs)
            raise ConfigError(
                f"Invalid configuration ({len(errs)} problem(s)):\n  - {bullet}"
            )
        return self


# ---------------------------------------------------------------------------
# Introspection (for --dry-run pretty print and a future GUI)
# ---------------------------------------------------------------------------
def describe() -> list[dict[str, Any]]:
    """Machine-readable schema description (one dict per field)."""
    return [
        {
            "name": f.name, "type": f.type.__name__, "default": f.default,
            "choices": list(f.choices) if f.choices else None,
            "is_list": f.is_list, "is_flag": f.is_flag, "optional": f.optional,
            "stage": f.stage, "help": f.help,
        }
        for f in FIELDS
    ]
