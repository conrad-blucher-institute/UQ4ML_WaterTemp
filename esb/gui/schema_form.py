"""Pure (streamlit-free) logic behind the GUI form.

Everything here is driven by the ``Field`` registry in ``esb.config`` — the
single source of truth. Add/remove a Field there and the widget plan, the
profile writer, and the profile reader all follow with NO edit to this file
or to ``app.py`` (that is the whole point; see design doc §8 and the
verification test in ``esb/tests/test_gui_schema.py``).

Kept separate from ``app.py`` so it is unit-testable without streamlit
installed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from esb.config import FIELDS, Field

# Widget kinds the app knows how to render. widget_kind() maps every possible
# Field shape to exactly one of these; anything unmapped raises loudly
# (guardrail 2 — never silently skip a schema field).
WIDGET_KINDS = (
    "checkbox",        # is_flag booleans
    "multiselect",     # list with fixed choices
    "list_text",       # free-form list (space-separated, typed items)
    "selectbox",       # scalar with fixed choices
    "number",          # scalar int/float
    "text",            # scalar str (paths etc.)
)


def widget_kind(f: Field) -> str:
    """Map one schema Field to the widget that renders it."""
    if f.is_flag:
        return "checkbox"
    if f.is_list:
        return "multiselect" if f.choices else "list_text"
    if f.choices:
        return "selectbox"
    if f.type in (int, float):
        return "number"
    if f.type is str:
        return "text"
    raise ValueError(
        f"schema field {f.name!r} has no widget mapping "
        f"(type={f.type.__name__}, is_list={f.is_list}, is_flag={f.is_flag}) — "
        f"extend widget_kind() in esb/gui/schema_form.py."
    )


def grouped_fields() -> list[tuple[str, list[Field]]]:
    """Fields grouped by ``Field.stage``, preserving schema order."""
    groups: dict[str, list[Field]] = {}
    for f in FIELDS:
        groups.setdefault(f.stage, []).append(f)
    return list(groups.items())


def parse_list_text(f: Field, text: str) -> list:
    """Parse a space/comma-separated list_text value with the field's item type."""
    items = text.replace(",", " ").split()
    try:
        return [f.type(tok) for tok in items]
    except ValueError as e:
        raise ValueError(f"{f.name}: could not parse {text!r} as {f.type.__name__} list ({e})")


# ---------------------------------------------------------------------------
# Profile writing (action a: "Build config & stop")
# ---------------------------------------------------------------------------
def profile_text(values: dict[str, Any], title: str = "") -> str:
    """Render a FULL resolved dump as a profiles/*.txt the CLI accepts unchanged.

    Every field appears, in schema order. Lines whose value equals the schema
    default carry an inline ``# default`` marker (the CLI's @file reader strips
    ``#`` comments, so these parse today with no argparse changes). Flags that
    are off and optionals that are None cannot be expressed as CLI tokens, so
    they are recorded as comment lines — still a complete, human-auditable
    record of the resolved config.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# {title or 'profile'} — built by esb gui, {ts}", "# Full resolved dump; '# default' marks values equal to the schema default."]
    for f in FIELDS:
        v = values[f.name]
        is_default = v == f.default
        mark = "  # default" if is_default else ""
        if f.is_flag:
            lines.append(f"{f.flag}{mark}" if v else f"# {f.flag}: off{mark}")
        elif v is None:
            lines.append(f"# {f.flag}: unset (None){mark}")
        elif f.is_list:
            lines.append(f"{f.flag} {' '.join(str(x) for x in v)}{mark}")
        else:
            sv = str(v)
            if any(c.isspace() for c in sv):
                # The @file reader is whitespace-tokenized; a value with spaces
                # would silently split. Fail loud (guardrail 2).
                raise ValueError(
                    f"{f.name}={sv!r} contains whitespace — profile files are "
                    f"whitespace-tokenized and cannot represent it. Use a "
                    f"path without spaces."
                )
            lines.append(f"{f.flag} {sv}{mark}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Profile reading (pre-fill the form from an existing profiles/*.txt)
# ---------------------------------------------------------------------------
def parse_profile(path: Path) -> dict[str, Any]:
    """Read a profiles/*.txt into a values dict (schema defaults fill gaps).

    Mirrors the CLI's @file semantics: whitespace-tokenized, ``#`` starts a
    comment. Arity comes from the schema (flag / list / scalar). Unknown flags
    raise loudly rather than being skipped.
    """
    from esb.config import FIELDS_BY_NAME

    tokens: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        tokens.extend(line.split())

    values = {f.name: f.default for f in FIELDS}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ValueError(f"{path}: unexpected token {tok!r} (expected a --flag)")
        name = tok[2:]
        f = FIELDS_BY_NAME.get(name)
        if f is None:
            raise ValueError(f"{path}: unknown flag {tok!r} (not in the Config schema)")
        if f.is_flag:
            values[name] = True
            i += 1
        elif f.is_list:
            j = i + 1
            items = []
            while j < len(tokens) and not tokens[j].startswith("--"):
                items.append(f.type(tokens[j]))
                j += 1
            if not items:
                raise ValueError(f"{path}: {tok} expects at least one value")
            values[name] = items
            i = j
        else:
            if i + 1 >= len(tokens):
                raise ValueError(f"{path}: {tok} expects a value")
            values[name] = f.type(tokens[i + 1])
            i += 2
    return values
