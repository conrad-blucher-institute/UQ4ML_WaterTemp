"""GUI single-source-of-truth proofs (pure python — no streamlit needed).

Load-bearing properties (design doc §8, Stage E Part 2 verification):
  1. EVERY schema Field maps to a widget — the form cannot silently drop one.
  2. Adding a Field to esb/config.py surfaces in the widget plan AND the
     profile writer with ZERO edits to any gui code (demonstrated live by
     appending a dummy Field and re-deriving both).
  3. A profile the GUI writes (full resolved dump, '# default' markers) parses
     back to the same values AND is accepted by the real CLI parser unchanged.

Run:  python -m pytest esb/tests/test_gui_schema.py -q
  or: python esb/tests/test_gui_schema.py   (plain runner, no pytest needed)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TUNE_DIR = _REPO_ROOT / "esb_03_2026_tune_experiment"
for _p in (str(_REPO_ROOT), str(_TUNE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from esb.config import FIELDS, FIELDS_BY_NAME, Config, Field  # noqa: E402
from esb.gui.schema_form import (  # noqa: E402
    WIDGET_KINDS, grouped_fields, parse_list_text, parse_profile,
    profile_text, widget_kind,
)


def _default_values() -> dict:
    return {f.name: f.default for f in FIELDS}


# ---------------------------------------------------------------------------
# 1. Coverage: every schema field renders; grouping preserves order
# ---------------------------------------------------------------------------
def test_every_field_has_a_widget():
    for f in FIELDS:
        kind = widget_kind(f)
        assert kind in WIDGET_KINDS, f"{f.name}: unknown widget kind {kind!r}"


def test_grouped_fields_cover_schema_in_order():
    flat = [f.name for _, fields in grouped_fields() for f in fields]
    assert sorted(flat) == sorted(f.name for f in FIELDS)
    for _, fields in grouped_fields():
        idx = [FIELDS.index(f) for f in fields]
        assert idx == sorted(idx), "stage group must preserve schema order"


# ---------------------------------------------------------------------------
# 2. Schema-change demo: dummy Field appears with zero gui-code edits
# ---------------------------------------------------------------------------
def test_new_schema_field_appears_without_gui_edit():
    dummy = Field("dummy_gui_probe", int, 7, "temporary probe field", stage="run")
    FIELDS.append(dummy)
    FIELDS_BY_NAME[dummy.name] = dummy
    try:
        assert widget_kind(dummy) == "number"
        assert any(f.name == "dummy_gui_probe"
                   for _, fs in grouped_fields() for f in fs)
        text = profile_text(_default_values() | {"dummy_gui_probe": 7})
        assert "--dummy_gui_probe 7  # default" in text
    finally:
        FIELDS.remove(dummy)
        del FIELDS_BY_NAME[dummy.name]
    assert not any(f.name == "dummy_gui_probe" for f in FIELDS)


# ---------------------------------------------------------------------------
# 3. Profile round-trips
# ---------------------------------------------------------------------------
def _roundtrip(values: dict) -> dict:
    text = profile_text(values, title="test")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(text)
        path = Path(fh.name)
    try:
        return parse_profile(path)
    finally:
        path.unlink()


def test_roundtrip_defaults():
    assert _roundtrip(_default_values()) == _default_values()


def test_roundtrip_non_defaults():
    values = _default_values() | {
        "loss": "mse", "scale": True, "lead_times": [12],
        "activations": ["relu", "selu"], "batch_size": 32,
        "shard": "3/40", "repetitions": 2, "learning_rate": 0.01,
    }
    back = _roundtrip(values)
    assert back == values
    Config(back).validate()  # accepted by the real validator unchanged


def test_default_markers_present():
    text = profile_text(_default_values())
    assert "# default" in text
    # a non-default line must NOT carry the marker
    text2 = profile_text(_default_values() | {"loss": "mse"})
    line = next(l for l in text2.splitlines() if l.startswith("--loss"))
    assert "# default" not in line


def test_whitespace_value_fails_loud():
    try:
        profile_text(_default_values() | {"output_dir": "C:/bad path/results"})
    except ValueError as e:
        assert "whitespace" in str(e)
    else:
        raise AssertionError("expected ValueError for whitespace path")


def test_parse_profile_rejects_unknown_flag():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("--no_such_flag 1\n")
        path = Path(fh.name)
    try:
        parse_profile(path)
    except ValueError as e:
        assert "unknown flag" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown flag")
    finally:
        path.unlink()


def test_parse_list_text_types_and_errors():
    lt = FIELDS_BY_NAME["num_layers"]
    assert parse_list_text(lt, "1 2, 3") == [1, 2, 3]
    try:
        parse_list_text(lt, "1 x")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# 3b. The written profile is accepted by the REAL CLI (@file --dry-run path)
# ---------------------------------------------------------------------------
def test_cli_accepts_gui_profile():
    from esb.cli import main as cli_main

    text = profile_text(_default_values() | {"lead_times": [12], "shard": "1/4"})
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(text)
        path = Path(fh.name)
    try:
        rc = cli_main(["run", f"@{path}", "--dry-run"])
        assert rc == 0
    finally:
        path.unlink()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
