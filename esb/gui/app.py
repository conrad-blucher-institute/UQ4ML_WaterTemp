"""esb GUI — schema-driven Streamlit app (design doc §8).

Launch:  python -m esb gui        (or: streamlit run esb/gui/app.py)

Rendering is 100% driven by ``esb.config.describe()``/``FIELDS`` via
``schema_form`` — this file contains NO hardcoded option names. Adding or
removing a Field in esb/config.py changes this form with zero edits here.

Three actions (design §8):
  a) Build config & stop — writes esb/profiles/<name>.txt (the dry-run artifact)
  b) Run locally         — shells out to `python -m esb run @profile` on THIS machine
  c) Open viz suite      — launches the existing viz runner on a results folder
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

# Make `import esb` work when launched via `streamlit run esb/gui/app.py`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esb.cli import PROFILES_DIR, _RUN_LOCATION_BANNER  # noqa: E402
from esb.config import Config, ConfigError, FIELDS_BY_NAME  # noqa: E402
from esb.gui.schema_form import (  # noqa: E402
    grouped_fields, parse_list_text, parse_profile, profile_text, widget_kind,
)

_VIZ_RUNNER = _REPO_ROOT / "esb_03_2026_tune_experiment" / "tuning-viz-folder" / "run_v16.py"

st.set_page_config(page_title="esb pipeline", layout="wide")
st.title("ESB water-temperature pipeline")
st.info(
    "**Where does this run?** \"Run locally\" trains on *this* machine. To run "
    "on the cluster (Grendal/SLURM), use \"Build config & stop\", then submit "
    "the generated `.txt` on the cluster manually. This is intentional — the "
    "GUI never submits remote jobs. Multi-machine sharding: run the CLI with a "
    "different `--shard k/N` on each machine yourself."
)

# ---------------------------------------------------------------------------
# Profile pre-fill (optional) — one reader shared with the CLI's semantics
# ---------------------------------------------------------------------------
existing = sorted(p.stem for p in PROFILES_DIR.glob("*.txt"))
with st.sidebar:
    st.header("Start from a profile")
    chosen = st.selectbox("Pre-fill form from", ["(schema defaults)"] + existing)
    if st.button("Load"):
        try:
            st.session_state["prefill"] = (
                {} if chosen == "(schema defaults)"
                else parse_profile(PROFILES_DIR / f"{chosen}.txt")
            )
            st.session_state["prefill_name"] = chosen
        except ValueError as e:
            st.error(str(e))

prefill: dict = st.session_state.get("prefill", {})


def _initial(f) -> object:
    return prefill.get(f.name, f.default)


# ---------------------------------------------------------------------------
# The schema-driven form (grouped by Field.stage; order = schema order)
# ---------------------------------------------------------------------------
values: dict[str, object] = {}
parse_errors: list[str] = []

for stage, fields in grouped_fields():
    with st.expander(f"{stage}", expanded=(stage in ("grid", "run"))):
        for f in fields:
            kind = widget_kind(f)
            init = _initial(f)
            key = f"fld_{f.name}"
            if f.optional:
                on = st.checkbox(f"set {f.name}?", value=init is not None,
                                 key=f"opt_{f.name}", help=f.help)
                if not on:
                    values[f.name] = None
                    continue
                if init is None:
                    init = f.type() if kind != "text" else ""
            if kind == "checkbox":
                values[f.name] = st.checkbox(f.name, value=bool(init), key=key, help=f.help)
            elif kind == "multiselect":
                values[f.name] = st.multiselect(
                    f.name, options=list(f.choices), default=list(init), key=key, help=f.help)
            elif kind == "list_text":
                raw = st.text_input(f.name, value=" ".join(str(x) for x in init),
                                    key=key, help=f.help + " (space-separated)")
                try:
                    values[f.name] = parse_list_text(f, raw)
                except ValueError as e:
                    parse_errors.append(str(e))
                    values[f.name] = list(init)
            elif kind == "selectbox":
                opts = list(f.choices)
                values[f.name] = st.selectbox(
                    f.name, options=opts,
                    index=opts.index(init) if init in opts else 0, key=key, help=f.help)
            elif kind == "number":
                step = 1 if f.type is int else 0.001
                values[f.name] = f.type(st.number_input(
                    f.name, value=f.type(init), step=step, format="%d" if f.type is int else "%.5f",
                    key=key, help=f.help))
            else:  # text
                values[f.name] = st.text_input(f.name, value=str(init), key=key, help=f.help)

# ---------------------------------------------------------------------------
# Live validation + job count (same code paths as the CLI)
# ---------------------------------------------------------------------------
config = Config(values)
problems = list(parse_errors)
try:
    config.validate()
except ConfigError as e:
    problems.append(str(e))

st.divider()
if problems:
    st.error("\n\n".join(problems))
else:
    from esb.pipeline import _build_grid_config  # light import (no TF)
    from esb.sharding import parse_shard, shard_bounds

    jobs = _build_grid_config(config).generate_configs()
    n = len(jobs)
    shard = parse_shard(values.get("shard"))
    if shard:
        s, e = shard_bounds(n, *shard)
        st.success(f"Valid ✓ — grid = {n} jobs; shard {shard[0]}/{shard[1]} runs "
                   f"jobs {s + 1}..{e} ({e - s} jobs) × {values['repetitions']} rep(s).")
    else:
        st.success(f"Valid ✓ — grid = {n} jobs × {values['repetitions']} rep(s).")

non_default = [f.name for f in FIELDS_BY_NAME.values() if values.get(f.name) != f.default]
st.caption("Changed from defaults: " + (", ".join(non_default) or "(none)"))

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.subheader("a) Build config & stop")
    default_name = st.session_state.get("prefill_name", "")
    name = st.text_input("Profile name", value="" if default_name.startswith("(") else default_name)
    if st.button("Write profile", disabled=bool(problems) or not name):
        path = PROFILES_DIR / f"{name}.txt"
        try:
            path.write_text(profile_text(values, title=name), encoding="utf-8")
            st.session_state["last_profile"] = str(path)
            st.success(f"Wrote {path}")
            st.code(f"python -m esb run --profile {name}", language="bash")
        except ValueError as e:
            st.error(str(e))

with col_b:
    st.subheader("b) Run locally")
    st.caption(_RUN_LOCATION_BANNER)
    last = st.session_state.get("last_profile")
    if st.button("Run on THIS machine", disabled=bool(problems) or not last):
        # Always run THROUGH the written profile file — one code path, and the
        # run is reproducible from the artifact.
        cmd = [sys.executable, "-m", "esb", "run", f"@{last}"]
        st.code(" ".join(cmd), language="bash")
        box = st.empty()
        buf: list[str] = []
        proc = subprocess.Popen(cmd, cwd=_REPO_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace")
        for line in proc.stdout:
            buf.append(line.rstrip())
            box.code("\n".join(buf[-40:]))
        proc.wait()
        (st.success if proc.returncode == 0 else st.error)(f"exit code {proc.returncode}")
    elif not last:
        st.caption("Write a profile first (action a).")

with col_c:
    st.subheader("c) Open viz suite")
    folder = st.text_input("Results folder (holds *_progress.csv)",
                           value=str(_REPO_ROOT / "results"))
    if st.button("Launch viz"):
        if not Path(folder).is_dir():
            st.error(f"Not a directory: {folder}")
        else:
            subprocess.Popen([sys.executable, str(_VIZ_RUNNER), "--folder", folder],
                             cwd=_VIZ_RUNNER.parent)
            st.success("Viz runner launched (serves plots in your browser).")
