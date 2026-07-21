"""Build or update the ESB update deck (.pptx) from docs/deck_spec.yaml.

The spec is the single source of content. The .pptx is edited IN PLACE when it
already exists: only the slides you target are touched, and within them only
the shapes this script created (named "esb:*"). Manual tweaks elsewhere — and
manual position/size changes to esb shapes — survive an update. Replacing a
shape's text does reset manual character formatting inside that shape.

Usage (run in the NLL_streamlit env; needs python-pptx + pyyaml):
    python scripts/make_deck.py                    # build new / update ALL slides
    python scripts/make_deck.py --slide 5 refs     # update only spec ids 5 and refs
    python scripts/make_deck.py --pull             # PowerPoint -> spec: copy text
                                                   # edits (title/bullets/table/notes)
                                                   # back into the yaml
    python scripts/make_deck.py --fresh            # discard deck, rebuild from spec
    python scripts/make_deck.py --theme tamu       # fresh build into a copy of
                                                   # scripts/deck_themes/tamu.pptx

Slides are matched to spec entries by the "esb:id:<id>" tag stored in each
slide's first shape name — reordering slides in PowerPoint does not break
updates. Missing images and unknown spec fields fail loud.

Two-way sync (automatic): text edits may be made EITHER in the yaml OR inside
the esb:* shapes in PowerPoint. A sidecar (docs/.<deck>.sync) records both
files' mtimes at every script write. On a default run:
  only yaml changed -> plain push;  only deck changed -> AUTO-PULL then push;
  both changed      -> fail loud (--pull keeps deck text, --force keeps yaml).
--fresh never pulls and refuses if the deck has un-pulled edits. Images never
pull — the spec's paths stay authoritative. Character formatting inside esb
shapes cannot round-trip (yaml holds plain text); polish once, at the end.

Examples:
  # Full refresh after new tuning results (run tests on grendal first):
  #   python scripts/tuning_tests/run_all_tests.py --shards 1 11 21 23 28 31 \
  #          --max-reps 10 --out-dir results/tuning_tests_phase1
  # then rebuild the deck (auto-pulls any PowerPoint text edits first):
  python scripts/make_deck.py

  # Refresh only the interim-results slides after a test rerun:
  python scripts/make_deck.py --slide 5 5b

  # I reworded things in PowerPoint and just want the yaml to catch up:
  python scripts/make_deck.py --pull

  # I edited BOTH the deck and the yaml (conflict): pick a winner
  python scripts/make_deck.py --pull    # deck text wins
  python scripts/make_deck.py --force   # yaml wins, deck edits discarded

  # Start over on a branded template:
  python scripts/make_deck.py --fresh --theme tamu
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPO_ROOT / "docs" / "deck_spec.yaml"
DEFAULT_OUT = REPO_ROOT / "docs" / "ESB_update_deck.pptx"
THEME_DIR = Path(__file__).resolve().parent / "deck_themes"

try:
    import yaml
except ImportError:
    sys.exit("ERROR: pyyaml not installed. Run: pip install pyyaml")
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
except ImportError:
    sys.exit("ERROR: python-pptx not installed. Run: pip install python-pptx")

# 16:9 geometry (EMU via Inches)
SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.5)
TITLE_H = Inches(0.9)
BODY_TOP = MARGIN + TITLE_H + Inches(0.1)
BODY_H = SLIDE_H - BODY_TOP - MARGIN

ALLOWED_KEYS = {"id", "title", "bullets", "image", "images", "table", "notes"}


# ---------------------------------------------------------------- spec

def _images_of(spec) -> list:
    """Normalized image list for a slide spec ('image' str or 'images' list)."""
    if spec.get("images"):
        return list(spec["images"])
    return [spec["image"]] if spec.get("image") else []


def load_spec(path: Path) -> list:
    if not path.exists():
        sys.exit(f"ERROR: spec not found: {path}")
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(spec, list) or not spec:
        sys.exit(f"ERROR: spec must be a non-empty list of slides: {path}")
    seen = set()
    for i, s in enumerate(spec, 1):
        unknown = set(s) - ALLOWED_KEYS
        if unknown:
            sys.exit(f"ERROR: slide #{i}: unknown field(s) {sorted(unknown)} "
                     f"(allowed: {sorted(ALLOWED_KEYS)})")
        for req in ("id", "title"):
            if not s.get(req):
                sys.exit(f"ERROR: slide #{i}: missing required field '{req}'")
        sid = str(s["id"])
        if sid in seen:
            sys.exit(f"ERROR: duplicate slide id '{sid}'")
        seen.add(sid)
        if s.get("image") and s.get("images"):
            sys.exit(f"ERROR: slide '{sid}': use 'image' OR 'images', not both")
        for rel in _images_of(s):
            img = REPO_ROOT / rel
            if not img.exists():
                if img.parent.exists():
                    pngs = sorted(p.name for p in img.parent.glob("*.png"))
                    hint = f"\n  PNGs present in {img.parent}: {pngs or 'none'}"
                else:
                    hint = f"\n  Directory does not exist: {img.parent}"
                sys.exit(f"ERROR: slide '{sid}': image not found: {img}{hint}\n"
                         "  Regenerate charts (scripts/tuning_tests/run_all_tests.py)"
                         " or fix the path in the spec.")
    return spec


# ---------------------------------------------------------- shape builders

def _tag(slide):
    """Return this slide's spec id, or None if not created by this script."""
    for shp in slide.shapes:
        if shp.name.startswith("esb:id:"):
            return shp.name[len("esb:id:"):]
    return None


def _find(slide, name):
    for shp in slide.shapes:
        if shp.name == name:
            return shp
    return None


def _fill_title(box, text):
    tf = box.text_frame
    tf.word_wrap = True
    tf.text = text
    p = tf.paragraphs[0]
    p.font.size = Pt(30)
    p.font.bold = True


def _fill_bullets(box, bullets):
    tf = box.text_frame
    tf.word_wrap = True
    tf.clear()
    for i, raw in enumerate(bullets):
        level = 1 if raw.startswith("  ") else 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if level == 0 else "– ") + raw.strip()
        p.level = level
        p.font.size = Pt(16 if level == 0 else 14)
        p.space_after = Pt(8)


def _add_image(slide, spec_img, left, top, max_w, max_h, name="esb:image:0"):
    pic = slide.shapes.add_picture(str(REPO_ROOT / spec_img), left, top,
                                   width=max_w)
    if pic.height > max_h:  # keep aspect, refit on the constraining axis
        pic.width = int(pic.width * max_h / pic.height)
        pic.height = max_h
        pic.left = left + (max_w - pic.width) // 2
    pic.name = name
    return pic


def _add_images(slide, paths, left, top, w, h):
    """Lay n images into the region: side by side if it's wide, stacked if
    it's the narrow right column."""
    n = len(paths)
    gap = Inches(0.15)
    for i, p in enumerate(paths):
        if w >= h:  # full-width region -> columns
            cell_w = (w - gap * (n - 1)) // n
            _add_image(slide, p, left + i * (cell_w + gap), top, cell_w, h,
                       name=f"esb:image:{i}")
        else:       # right column -> rows
            cell_h = (h - gap * (n - 1)) // n
            _add_image(slide, p, left, top + i * (cell_h + gap), w, cell_h,
                       name=f"esb:image:{i}")


def _add_table(slide, rows, left, top, width):
    n_r, n_c = len(rows), len(rows[0])
    height = Inches(0.35) * n_r
    frame = slide.shapes.add_table(n_r, n_c, left, top, width, height)
    frame.name = "esb:table"
    tbl = frame.table
    for r, row in enumerate(rows):
        if len(row) != n_c:
            sys.exit(f"ERROR: table row {r + 1} has {len(row)} cells, "
                     f"header has {n_c}")
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(13)
                p.font.bold = (r == 0)
    return frame


def _rerun_commands(spec) -> list:
    """command.txt lines from the image folders (written by run_all_tests.py):
    the exact command that produced this slide's data, for the notes header."""
    cmds = []
    for rel in _images_of(spec):
        img_dir = (REPO_ROOT / rel).parent
        for probe in (img_dir / "command.txt", img_dir.parent / "command.txt"):
            if probe.exists():
                for line in probe.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and line not in cmds:
                        cmds.append(line)
                break
    return cmds


def _notes_text(spec):
    """Presenter notes = reference header (slide id/title + exact shape names
    and image files, so shapes can be named precisely when iterating; plus the
    recorded rerun command for the slide's data), a blank line, then the
    spec's notes prose."""
    names = [f"esb:id:{spec['id']} (title)"]
    if spec.get("table"):
        names.append("esb:table")
    if spec.get("bullets"):
        names.append("esb:body")
    for i, p in enumerate(_images_of(spec)):
        names.append(f"esb:image:{i} ({Path(p).name})")
    header = f"[slide id {spec['id']}] {spec['title']}\nshapes: " + ", ".join(names)
    for cmd in _rerun_commands(spec):
        header += f"\nrerun-to-refresh: {cmd}"
    prose = (spec.get("notes") or "").strip()
    return header + ("\n\n" + prose if prose else "")


def _set_notes(slide, text):
    tf = slide.notes_slide.notes_text_frame
    if tf is None:
        # The deck's notes master has no body placeholder (PowerPoint/theme
        # can strip it); python-pptx cannot create one. Slide content is
        # unaffected — only the presenter notes are skipped.
        sid = _tag(slide) or "?"
        print(f"  WARNING: slide id {sid}: could not write presenter notes "
              "(notes master lacks a text placeholder)")
        return
    tf.text = text.strip()


# ----------------------------------------------------------------- sync

def _sync_path(out: Path) -> Path:
    return out.parent / f".{out.name}.sync"


def _record_sync(out: Path, spec_path: Path):
    _sync_path(out).write_text(
        f"deck={out.stat().st_mtime_ns}\nspec={spec_path.stat().st_mtime_ns}\n",
        encoding="ascii")


def _sync_state(out: Path, spec_path: Path):
    """Return (deck_changed, spec_changed) since the last script sync.
    No sidecar yet -> (None, None): unknown, caller must not guess."""
    sync = _sync_path(out)
    if not sync.exists():
        return None, None
    rec = dict(line.split("=", 1) for line in
               sync.read_text(encoding="ascii").split() if "=" in line)
    return (rec.get("deck") != str(out.stat().st_mtime_ns),
            rec.get("spec") != str(spec_path.stat().st_mtime_ns))


# ----------------------------------------------------------------- pull

def _pull_bullets(body) -> list:
    out = []
    for p in body.text_frame.paragraphs:
        t = p.text
        for pref in ("• ", "– ", "•", "–"):
            if t.startswith(pref):
                t = t[len(pref):]
                break
        t = t.strip()
        if t:
            out.append(("  " if p.level else "") + t)
    return out


def _pull_notes(slide) -> str:
    if not slide.has_notes_slide:
        return ""
    tf = slide.notes_slide.notes_text_frame
    if tf is None:
        return ""
    text = tf.text
    if text.startswith("[slide id"):  # strip the generated reference header
        parts = text.split("\n\n", 1)
        text = parts[1] if len(parts) == 2 else ""
    return text.strip()


def pull_spec(prs, spec, targets):
    """Copy deck text (title/bullets/table/notes) back into spec entries."""
    by_id = {}
    for slide in prs.slides:
        tag = _tag(slide)
        if tag is not None:
            by_id[tag] = slide
    pulled, missing = [], []
    for entry in spec:
        sid = str(entry["id"])
        if sid not in targets:
            continue
        slide = by_id.get(sid)
        if slide is None:
            missing.append(sid)
            continue
        title = _find(slide, f"esb:id:{sid}")
        if title is not None and title.has_text_frame:
            entry["title"] = title.text_frame.text.strip()
        body = _find(slide, "esb:body")
        if body is not None:
            entry["bullets"] = _pull_bullets(body)
        tbl = _find(slide, "esb:table")
        if tbl is not None:
            entry["table"] = [[c.text for c in r.cells] for r in tbl.table.rows]
        notes = _pull_notes(slide)
        if notes:
            entry["notes"] = notes
        else:
            entry.pop("notes", None)
        pulled.append(sid)
    return pulled, missing


_KEY_ORDER = ["id", "title", "bullets", "image", "images", "table", "notes"]


def write_spec(path: Path, spec):
    """Rewrite the yaml: keep the leading comment header, dump slides below."""
    header = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("- "):
            break
        header.append(line)

    class _Dumper(yaml.SafeDumper):
        pass

    def _str(dumper, s):
        style = "|" if "\n" in s else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", s, style=style)

    _Dumper.add_representer(str, _str)
    ordered = [{k: e[k] for k in _KEY_ORDER if k in e} for e in spec]
    body = yaml.dump(ordered, Dumper=_Dumper, sort_keys=False,
                     allow_unicode=True, default_flow_style=False, width=1000)
    path.write_text("".join(header) + body, encoding="utf-8")


# ------------------------------------------------------------ build/update

def _layout(spec):
    """Compute body-region boxes: (bullets_box, table_box, image_box)."""
    has_b, has_t = (bool(spec.get(k)) for k in ("bullets", "table"))
    has_i = bool(_images_of(spec))
    full_w = SLIDE_W - 2 * MARGIN
    if has_i and (has_b or has_t):
        text_w = Inches(7.3)
        img_left = MARGIN + text_w + Inches(0.2)
        img_w = SLIDE_W - MARGIN - img_left
        return text_w, (img_left, img_w)
    return full_w, (MARGIN, full_w)


def build_slide(prs, spec):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # 6 = blank
    title = slide.shapes.add_textbox(MARGIN, MARGIN, SLIDE_W - 2 * MARGIN, TITLE_H)
    title.name = f"esb:id:{spec['id']}"  # id tag rides on the title shape
    _fill_title(title, spec["title"])

    text_w, (img_left, img_w) = _layout(spec)
    top = BODY_TOP
    if spec.get("table"):
        frame = _add_table(slide, spec["table"], MARGIN, top, text_w)
        top = frame.top + frame.height + Inches(0.2)
    if spec.get("bullets"):
        body = slide.shapes.add_textbox(MARGIN, top, text_w, SLIDE_H - top - MARGIN)
        body.name = "esb:body"
        _fill_bullets(body, spec["bullets"])
    imgs = _images_of(spec)
    if imgs:
        _add_images(slide, imgs, img_left, BODY_TOP, img_w, BODY_H)
    _set_notes(slide, _notes_text(spec))
    return slide


def update_slide(slide, spec):
    """Replace content of esb-named shapes in place; keep user geometry."""
    title = _find(slide, f"esb:id:{spec['id']}")
    _fill_title(title, spec["title"])

    body = _find(slide, "esb:body")
    if spec.get("bullets"):
        if body is None:
            text_w, _ = _layout(spec)
            body = slide.shapes.add_textbox(MARGIN, BODY_TOP, text_w,
                                            SLIDE_H - BODY_TOP - MARGIN)
            body.name = "esb:body"
        _fill_bullets(body, spec["bullets"])
    elif body is not None:
        body._element.getparent().remove(body._element)

    old_tbl = _find(slide, "esb:table")
    if old_tbl is not None:
        pos = (old_tbl.left, old_tbl.top, old_tbl.width)
        old_tbl._element.getparent().remove(old_tbl._element)
    else:
        text_w, _ = _layout(spec)
        pos = (MARGIN, BODY_TOP, text_w)
    if spec.get("table"):
        _add_table(slide, spec["table"], *pos)

    old_imgs = sorted((s for s in slide.shapes
                       if s.name.startswith("esb:image")), key=lambda s: s.name)
    geos = [(s.left, s.top, s.width, s.height) for s in old_imgs]
    for s in old_imgs:
        s._element.getparent().remove(s._element)
    imgs = _images_of(spec)
    if imgs:
        if len(geos) == len(imgs):  # same count: reuse user-adjusted geometry
            for i, (p, g) in enumerate(zip(imgs, geos)):
                _add_image(slide, p, *g, name=f"esb:image:{i}")
        else:
            _, (img_left, img_w) = _layout(spec)
            _add_images(slide, imgs, img_left, BODY_TOP, img_w, BODY_H)

    _set_notes(slide, _notes_text(spec))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--slide", nargs="+", metavar="ID", default=None,
                    help="update only these spec ids (default: all)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing deck, rebuild everything from spec")
    ap.add_argument("--theme", default=None,
                    help=f"template name in {THEME_DIR} (fresh builds only)")
    ap.add_argument("--pull", action="store_true",
                    help="reverse direction: copy PowerPoint text edits "
                         "(title/bullets/table/notes of esb:* shapes) into the spec")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the deck even if it has un-pulled "
                         "PowerPoint edits (DISCARDS them)")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    ids = [str(s["id"]) for s in spec]
    if args.slide:
        bad = [i for i in args.slide if i not in ids]
        if bad:
            sys.exit(f"ERROR: unknown slide id(s) {bad}. Spec has: {ids}")

    def save(prs):
        try:
            prs.save(str(args.out))
        except PermissionError:
            sys.exit(f"ERROR: can't write {args.out} — it is almost certainly "
                     "open in PowerPoint.\n  Close the deck there and rerun "
                     "(nothing was lost; the file is unchanged).")
        _record_sync(args.out, args.spec)

    def do_pull(prs, targets):
        pulled, missing = pull_spec(prs, spec, targets)
        write_spec(args.spec, spec)
        _record_sync(args.out, args.spec)
        print(f"Pulled PowerPoint text edits -> {args.spec}")
        print(f"  slides pulled: {pulled or 'none'}")
        if missing:
            print(f"  spec ids not found in deck (skipped): {missing}")
        print("  note: images and character formatting do NOT pull; "
              "spec image paths stay authoritative.")

    if args.pull:
        if args.fresh or args.theme:
            sys.exit("ERROR: --pull cannot combine with --fresh/--theme.")
        if not args.out.exists():
            sys.exit(f"ERROR: no deck to pull from: {args.out}")
        do_pull(Presentation(str(args.out)), args.slide or ids)
        return

    fresh = args.fresh or not args.out.exists()
    if args.theme and not fresh:
        sys.exit("ERROR: --theme applies to fresh builds only (add --fresh).")

    auto_pull = False
    if args.out.exists() and not args.force:
        deck_changed, spec_changed = _sync_state(args.out, args.spec)
        if deck_changed is None:
            sys.exit(f"ERROR: {args.out.name} exists but has never been synced "
                     "(no sidecar) — it may hold PowerPoint edits the spec lacks.\n"
                     "  python scripts/make_deck.py --pull   # adopt deck text into the spec\n"
                     "  --force                              # trust the spec, overwrite the deck")
        if deck_changed and spec_changed:
            sys.exit("ERROR: BOTH the deck (PowerPoint edits) and the spec (yaml "
                     "edits) changed since the last sync — cannot merge.\n"
                     "  python scripts/make_deck.py --pull   # keep the DECK's text (overwrites yaml text)\n"
                     "  --force                              # keep the YAML (discards deck edits)")
        auto_pull = deck_changed and not fresh
        if deck_changed and fresh:
            sys.exit("ERROR: the deck has un-pulled PowerPoint edits; --fresh "
                     "would discard them.\n  Run --pull first, or add --force.")

    if auto_pull:
        print("Deck edited in PowerPoint since last sync -> auto-pulling "
              "text into the spec before pushing...")
        do_pull(Presentation(str(args.out)), ids)

    if fresh:
        if args.theme:
            tpl = THEME_DIR / f"{args.theme}.pptx"
            if not tpl.exists():
                have = sorted(p.stem for p in THEME_DIR.glob("*.pptx")) \
                    if THEME_DIR.exists() else []
                sys.exit(f"ERROR: theme not found: {tpl}\n  Available: {have}")
            prs = Presentation(str(tpl))
        else:
            prs = Presentation()
            prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
        for s in spec:
            build_slide(prs, s)
        save(prs)
        print(f"Built fresh deck: {len(spec)} slides -> {args.out}")
        return

    # in-place update
    prs = Presentation(str(args.out))
    by_id = {}
    untagged = 0
    for slide in prs.slides:
        tag = _tag(slide)
        if tag is None:
            untagged += 1
        else:
            by_id[tag] = slide

    targets = args.slide or ids
    updated, added = [], []
    for s in spec:
        sid = str(s["id"])
        if sid not in targets:
            continue
        if sid in by_id:
            update_slide(by_id[sid], s)
            updated.append(sid)
        else:
            build_slide(prs, s)  # appended at end; reorder in PowerPoint
            added.append(sid)

    stale = sorted(set(by_id) - set(ids))
    save(prs)
    print(f"Updated in place -> {args.out}")
    print(f"  updated: {updated or 'none'}")
    if added:
        print(f"  added at end (reorder manually): {added}")
    if stale:
        print(f"  WARNING: deck has slides for ids no longer in spec: {stale} "
              "(python-pptx can't delete; remove them in PowerPoint)")
    if untagged:
        print(f"  note: {untagged} manually-added slide(s) left untouched")


if __name__ == "__main__":
    main()
