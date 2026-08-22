"""Bake the wide prediction CSVs into a self-contained cove_timeseries.html.

This is COVE's time-series page - the tab the explorer's "Send to Time Series"
button opens. It carries its own data and its own copy of plotly.js, so it opens
straight from disk: no server, no `serve_timeseries_v2.py`, no python on the
recipient's machine. That is the whole point - the old 05_timeseries.html
fetched per-model prediction CSVs from a local server, which nobody who receives
the file by email can run.

Handoff from the explorer
-------------------------
The explorer writes its scatter selection to ``localStorage['tsModels']``; this
page reads it on load and pre-ticks those combos. Browsers treat all file:// pages
as one origin, so the handoff works with both files simply sitting in a folder.

Opening the page with NO selection is normal and lands on a blank chart with the
pickers ready - the explorer deliberately opens it that way when nothing is
selected.

Traces are OFF by default
-------------------------
360 curves drawn at once is unreadable and slow. Nothing is plotted until you
pick something: choose a lead time to bring in every combo at that lead time, or
tick an individual combo. The observation line is the one exception - it is the
reference every curve is read against, so it is always on.

Each combo contributes N repetition traces plus a mean-of-N trace. N is read
from the data, never assumed to be 10.

Favorites and notes come from the selection CSV
-----------------------------------------------
``modeler_favorite`` rows draw DASHED, get a star in the legend, and are added
last so they sit on top of their siblings. ``notes`` appear in the hover and in
a right-rail panel listing every currently-visible combo, so the reasoning
travels with the chart instead of living in a separate spreadsheet.

Where this lives
----------------
This is one of COVE's plots, so it sits with the others in ``plot_types/`` and is
drawn by the runner in the same pass - not as a separate script you have to
remember. Its input is the wide prediction CSVs produced by ``python -m
esb.predict``, which is a heavyweight TF step run only when the models change.

One page, or one per lead time
------------------------------
A single page carrying every lead time is right for a hand-picked selection. It
is not right for a campaign holding the top 20 combos at all 21 lead times: that
is ~16,800 models, and one page would be gigabytes of curves of which any one
reading uses a twentieth.

When ``esb.predict --split-by-leadtime`` has left ``predictions_wide/LT<n>/``
subfolders, this writes ``cove_timeseries_LT<n>.html`` per lead time instead -
detected from the layout on disk, not from a second flag that could disagree
with it. ``--single-page`` / ``--split-by-leadtime`` override the detection.

Usage
-----
Normally: just run the suite (``run_v16.py --folder <campaign>``).
Standalone, for iterating on the page itself:

    python tuning_viz/plot_types/timeseries_cove.py results/phase3_selected
    python tuning_viz/plot_types/timeseries_cove.py <folder> --datasets 2021
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# Repo root: plot_types -> tuning_viz -> tuning-viz-folder -> pkg -> repo
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Plotly slows to a crawl long before the browser runs out of memory, and a
# too-large bake is discovered by the RECIPIENT, not by you. Fail at build time.
#
# 800 was sized for a hand-picked selection. A per-lead-time page over the top-20
# campaign draws 20 combos x 4 cycles x (10 reps + 1 mean) = 880 for a single
# dataset, so the cap has to clear that with room to spare - it is a guard
# against an unusable page, not a budget to be tuned against.
DEFAULT_MAX_TRACES = 2000


def _norm_act(value: str) -> str:
    return str(value).strip().lower()


def _norm_num(value) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def combo_key(lead_time, cycle, activation, num_layers, neurons, dropout):
    return (_norm_num(lead_time), _norm_num(cycle), _norm_act(activation),
            _norm_num(num_layers), _norm_num(neurons), _norm_num(dropout))


def key_string(key) -> str:
    """Stable join of a combo key, formatted the way JavaScript formats numbers.

    The page matches the explorer's selection against this string, and JS renders
    3.0 as "3". Using python's str() would emit "3.0" and nothing would ever
    match - a silent empty chart. ``%g`` produces the same text as JS String().
    """
    lt, cy, act, layers, neurons, drop = key
    return "|".join([f"{lt:g}", f"{cy:g}", act, f"{layers:g}",
                     f"{neurons:g}", f"{drop:g}"])


def combo_label(key) -> str:
    lt, cy, act, layers, neurons, drop = key
    return (f"LT{lt:g}h cycle{cy:g} {act} {layers:g}L {neurons:g}N "
            f"d{drop:g}")


def load_selection(path: Path) -> dict[tuple, dict]:
    """Map combo key -> {'favorite': bool, 'notes': str} from the tidy CSV."""
    if not path.exists():
        print(f"  note: no selection CSV at {path} (no favorites/notes)")
        return {}
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            key = combo_key(row.get("leadtime"), row.get("cycle"),
                            row.get("activation"), row.get("layers"),
                            row.get("neurons"), row.get("dropout"))
            fav = str(row.get("modeler_favorite", "")).strip().lower()
            out[key] = {
                "favorite": fav in ("true", "1", "yes", "y"),
                "notes": (row.get("notes") or "").strip(),
            }
    return out


def build_payload(folder: Path, datasets: list[str], round_to: int,
                  wide_dir: Path | None = None, by_name: dict | None = None,
                  selection: dict | None = None):
    """Return the JSON payload embedded in the page.

    ``wide_dir`` is where the ``*_wide.csv`` live; it defaults to the campaign's
    ``predictions_wide/`` but points at a ``predictions_wide/LT<n>/`` subfolder
    when the pages are split by lead time.

    ``by_name`` and ``selection`` are passed in when baking many pages from one
    campaign, so the (slow) campaign load and selection parse happen once for all
    21 pages instead of once per page.
    """
    if by_name is None:
        from esb.campaign import load_campaign
        by_name = {m["base_name"]: m for m in load_campaign(folder)}
    if selection is None:
        selection = load_selection(folder / "phase2_selection_for_phase3_tidy.csv")
    if wide_dir is None:
        wide_dir = folder / "predictions_wide"

    payload = {"datasets": {}, "combos": {}, "order": []}
    combos: dict[tuple, dict] = {}

    for dataset in datasets:
        wide_path = wide_dir / f"{dataset}_wide.csv"
        if not wide_path.exists():
            raise FileNotFoundError(
                f"missing {wide_path} - run `python -m esb.predict` first")

        with open(wide_path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            rows = list(reader)

        times = [r[0] for r in rows]
        obs_idx = header.index("observation")
        observation = [_clean(r[obs_idx], round_to) for r in rows]
        payload["datasets"][dataset] = {"x": times, "observation": observation}

        for col_idx, name in enumerate(header):
            if col_idx == 0 or name == "observation":
                continue
            model = by_name.get(name)
            if model is None:
                print(f"  WARNING: column {name} has no model in the campaign")
                continue
            key = combo_key(model["lead_time"], model["cycle"],
                            model["activation"], model["num_layers"],
                            model["neurons"], model["dropout"])
            combo = combos.setdefault(key, {
                "label": combo_label(key),
                "key": key_string(key),
                "lead_time": key[0],
                "reps": defaultdict(dict),
                "meta": selection.get(key, {"favorite": False, "notes": ""}),
            })
            series = [_clean(r[col_idx], round_to) for r in rows]
            combo["reps"][str(model.get("run_num"))][dataset] = series

    # Mean across whatever reps exist, per dataset. N is measured, not assumed.
    for key, combo in combos.items():
        combo["mean"] = {}
        for dataset in datasets:
            stacks = [rep[dataset] for rep in combo["reps"].values()
                      if dataset in rep]
            if not stacks:
                continue
            length = len(stacks[0])
            mean = []
            for i in range(length):
                vals = [s[i] for s in stacks if s[i] is not None]
                mean.append(round(sum(vals) / len(vals), round_to) if vals else None)
            combo["mean"][dataset] = mean
        combo["n_reps"] = len(combo["reps"])
        combo["reps"] = dict(combo["reps"])

    # Favorites LAST so they draw on top of their siblings.
    order = sorted(combos, key=lambda k: (combos[k]["meta"]["favorite"], k))
    payload["combos"] = {combo_label(k): combos[k] for k in order}
    payload["order"] = [combo_label(k) for k in order]
    payload["lead_times"] = sorted({combos[k]["lead_time"] for k in combos})
    return payload


def _clean(text: str, round_to: int):
    """CSV cell -> float or None. Blank means 'not evaluated here' (ragged join)."""
    if text is None or text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return None if math.isnan(value) else round(value, round_to)


PAGE = """<meta charset="utf-8">
<title>COVE Time Series</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1a1a1a; --muted: #666; --line: #d8d8d8;
    --panel: #f6f6f6; --accent: #2a6fb5;
  }
  /* Light is the default, deliberately: this page is read next to the other
     COVE pages and printed/screenshotted for slides, so it should not flip
     appearance with the reader's OS setting. Dark is opt-in only. */
  :root[data-theme="dark"] {
    --bg: #16181c; --fg: #e8e8e8; --muted: #9aa0a6; --line: #33373d;
    --panel: #1e2126; --accent: #6aa9e9;
  }
  body { background: var(--bg); color: var(--fg); margin: 0;
         font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }
  header { padding: 12px 16px; border-bottom: 1px solid var(--line); }
  h1 { font-size: 16px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 12px; }
  .wrap { display: flex; gap: 12px; padding: 12px; align-items: flex-start; }
  .controls { flex: 0 0 240px; }
  /* Keep the chart in a lower layer and stop it bleeding past its own box. */
  .chart { flex: 1 1 auto; min-width: 0; position: relative; z-index: 1;
           overflow: hidden; }
  /* position+z-index so the notes column sits ABOVE the chart. Plotly's SVG
     establishes its own stacking context and was painting over the rail, which
     made the notes look transparent. The background is a solid token, not a
     tint, so nothing shows through. */
  .rail { flex: 0 0 260px; max-height: 78vh; overflow-y: auto;
          background: var(--panel); border: 1px solid var(--line);
          padding: 10px;
          /* Heavier grey edge down the whole left side of the OUTER panel, so
             the notes column separates cleanly from the chart beside it. Square
             left corners keep it a crisp full-height line; the rounded corners
             were softening it into near-invisibility. */
          border-left: 4px solid #9aa0a6;
          border-radius: 0 6px 6px 0;
          position: relative; z-index: 100; opacity: 1; }
  fieldset { border: 1px solid var(--line); border-radius: 6px;
             margin: 0 0 10px; padding: 8px; }
  legend { color: var(--muted); font-size: 12px; padding: 0 4px; }
  .list { max-height: 200px; overflow-y: auto; }
  label { display: block; font-size: 12px; padding: 1px 0; cursor: pointer; }
  button { font: inherit; font-size: 12px; padding: 3px 8px; cursor: pointer;
           background: var(--bg); color: var(--fg);
           border: 1px solid var(--line); border-radius: 4px; }
  .note { border-bottom: 1px solid var(--line); padding: 6px 0; font-size: 12px; }
  .note b { display: block; }
  .note span { color: var(--muted); }
  .empty { color: var(--muted); font-size: 12px; }
</style>

<header>
  <h1>COVE &mdash; Time Series</h1>
  <div class="sub" id="summary"></div>
  <div class="sub" id="handoff" style="margin-top:4px"></div>
</header>

<div class="wrap">
  <div class="controls">
    <fieldset>
      <legend>Dataset</legend>
      <select id="dataset"></select>
    </fieldset>
    <fieldset>
      <legend>Lead time (selects the whole group)</legend>
      <div style="margin-bottom:6px">
        <button id="ltAll">Select all</button>
        <button id="ltNone">Select none</button>
      </div>
      <div class="list" id="leadtimes"></div>
    </fieldset>
    <fieldset>
      <legend>Individual combos</legend>
      <div style="margin-bottom:6px"><button id="clear">Clear all</button></div>
      <div class="list" id="combos"></div>
    </fieldset>
    <fieldset>
      <legend>Reps</legend>
      <label><input type="checkbox" id="showReps" checked> show individual reps</label>
      <label><input type="checkbox" id="showMean" checked> show mean</label>
      <label><input type="checkbox" id="soloReps"> toggle each rep separately</label>
    </fieldset>
    <fieldset>
      <legend>Notes</legend>
      <label><input type="checkbox" id="showNotes"> show notes column</label>
    </fieldset>
  </div>

  <div class="chart"><div id="plot" style="height:78vh"></div></div>

  <!-- Hidden by default: most combos carry no note, so an empty column would
       just eat chart width. Toggled from the Notes box on the left. -->
  <div class="rail" id="rail" style="display:none">
    <b style="font-size:12px">Notes (visible combos)</b>
    <div id="notes"></div>
  </div>
</div>

<script>
const DATA = __PAYLOAD__;
const MAX_TRACES = __MAX_TRACES__;
const selected = new Set();
let dataset = Object.keys(DATA.datasets)[0];

const $ = id => document.getElementById(id);
const palette = ['#2a6fb5','#c8552b','#3f8f4a','#8a5cc4','#b58b1f','#2f9c9c',
                 '#b4478f','#6b7280','#1f6f8b','#9c5a2f'];

function comboColor(label) {
  let h = 0;
  for (const ch of label) h = (h * 31 + ch.charCodeAt(0)) % 100000;
  return palette[h % palette.length];
}

function build() {
  const ds = DATA.datasets[dataset];
  const traces = [{
    x: ds.x, y: ds.observation, name: 'observation', mode: 'lines',
    line: { color: '#888', width: 2 }, hovertemplate: 'obs %{y:.2f}C<extra></extra>'
  }];

  // DATA.order already puts favorites last, so they are drawn on top.
  const solo = $('soloReps').checked;
  // With a single combo on screen there is no combo-vs-combo comparison
  // left to encode, so the shared colour is free for a more useful job:
  // telling the reps apart. With several combos selected the reps MUST
  // stay one colour per combo, otherwise the band that shows the spread
  // stops reading as a band.
  const colorByRep = selected.size === 1;
  for (const label of DATA.order) {
    if (!selected.has(label)) continue;
    const combo = DATA.combos[label];
    const color = comboColor(label);
    const fav = combo.meta.favorite;
    const nameBase = (fav ? '* ' : '') + label;
    const noteText = combo.meta.notes
      ? '<br><i>' + combo.meta.notes.replace(/</g, '&lt;') + '</i>' : '';

    if ($('showReps').checked) {
      for (const [run, byDs] of Object.entries(combo.reps)) {
        if (!byDs[dataset]) continue;
        // Each rep is its own visible, legend-listed trace: the point of showing
        // reps is to read the spread between them, so they are drawn solid and
        // only lightly faded. The mean is what gets the heavier line below.
        // run numbers start at 1; the palette wraps if a campaign ever has
        // more reps than colours.
        const repColor = colorByRep
          ? palette[(Number(run) - 1) % palette.length] : color;
        traces.push({
          x: ds.x, y: byDs[dataset], name: nameBase + ' run' + run,
          // Distinct colours are worth seeing clearly; faded overlapping lines
          // of one colour are not.
          mode: 'lines', opacity: colorByRep ? 0.9 : 0.55,
          // legendgroup ties every rep + the mean of one combo together, so a
          // legend click toggles the whole combo. Solo mode gives each trace
          // its OWN group, which is what makes single reps individually
          // clickable (and double-click-to-isolate work per rep) -- needed to
          // judge one repetition against the other four.
          legendgroup: solo ? nameBase + ' run' + run : label,
          line: { color: repColor, width: colorByRep ? 1.5 : 1 },
          hovertemplate: '%{y:.2f}C  run' + run + '<extra>' + label + '</extra>'
        });
      }
    }
    if ($('showMean').checked && combo.mean[dataset]) {
      traces.push({
        x: ds.x, y: combo.mean[dataset],
        name: nameBase + '  (mean of ' + combo.n_reps + ')',
        // Favorites stay solid. They are still called out by the star in the
        // legend, a heavier line, and being drawn last (on top).
        mode: 'lines',
        legendgroup: solo ? nameBase + ' mean' : label,
        line: { color: color, width: fav ? 3 : 2 },
        hovertemplate: '%{y:.2f}C<extra>' + label
          + '<br>mean of ' + combo.n_reps + ' reps' + noteText + '</extra>'
      });
    }
  }

  if (traces.length > MAX_TRACES) {
    alert('That selection is ' + traces.length + ' traces (limit ' + MAX_TRACES
      + '). Narrowing it keeps the chart usable.');
    return;
  }

  Plotly.react('plot', traces, {
    margin: { l: 50, r: 10, t: 10, b: 40 },
    xaxis: { title: 'target time', gridcolor: 'rgba(128,128,128,0.2)' },
    yaxis: { title: 'water temperature (C)', gridcolor: 'rgba(128,128,128,0.2)' },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: getComputedStyle(document.body).color },
    hovermode: 'closest', showlegend: true,
    legend: { orientation: 'h', y: -0.15 },
    // Keep the current zoom/pan when traces are added or removed. Plotly resets
    // the axes on every react() unless uirevision stays the SAME value between
    // calls; holding it constant tells plotly the user's view still applies.
    // Changing dataset deliberately changes it, since the x-range from one
    // dataset is meaningless in another.
    uirevision: dataset
  }, { responsive: true });

  renderNotes();
  $('summary').textContent = selected.size + ' of ' + DATA.order.length
    + ' combos shown  -  ' + (traces.length - 1) + ' model traces  -  dataset: ' + dataset;
}

function toggleNotes() {
  const on = $('showNotes').checked;
  $('rail').style.display = on ? '' : 'none';
  if (on) renderNotes();
}

function renderNotes() {
  // Cheap guard: nothing to draw while the column is hidden.
  if (!$('showNotes').checked) return;
  const box = $('notes');
  box.innerHTML = '';
  const withNotes = DATA.order.filter(l => selected.has(l));
  if (!withNotes.length) {
    box.innerHTML = '<div class="empty">Nothing selected yet. Pick a lead time '
      + 'or a combo on the left.</div>';
    return;
  }
  for (const label of withNotes) {
    const combo = DATA.combos[label];
    const div = document.createElement('div');
    div.className = 'note';
    div.innerHTML = '<b>' + (combo.meta.favorite ? '* ' : '') + label + '</b>'
      + '<span>' + (combo.meta.notes || 'no notes') + '</span>';
    box.appendChild(div);
  }
}

function init() {
  const dsSel = $('dataset');
  for (const name of Object.keys(DATA.datasets)) {
    const opt = document.createElement('option');
    opt.value = name; opt.textContent = name;
    dsSel.appendChild(opt);
  }
  dsSel.onchange = e => { dataset = e.target.value; build(); };

  for (const lt of DATA.lead_times) {
    const label = document.createElement('label');
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.onchange = e => {
      // A lead time is a shortcut for its whole group of combos.
      for (const name of DATA.order) {
        if (DATA.combos[name].lead_time !== lt) continue;
        if (e.target.checked) selected.add(name); else selected.delete(name);
      }
      syncCombos(); build();
    };
    label.appendChild(box);
    label.appendChild(document.createTextNode(' LT ' + lt + 'h'));
    $('leadtimes').appendChild(label);
  }

  for (const name of DATA.order) {
    const combo = DATA.combos[name];
    const label = document.createElement('label');
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.dataset.combo = name;
    box.onchange = e => {
      if (e.target.checked) selected.add(name); else selected.delete(name);
      build();
    };
    label.appendChild(box);
    label.appendChild(document.createTextNode(
      ' ' + (combo.meta.favorite ? '* ' : '') + name));
    $('combos').appendChild(label);
  }

  $('ltAll').onclick = () => { setAllLeadTimes(true); };
  $('ltNone').onclick = () => { setAllLeadTimes(false); };
  $('clear').onclick = () => { selected.clear(); syncCombos(); build(); };
  $('showReps').onchange = build;
  $('showMean').onchange = build;
  $('soloReps').onchange = build;
  $('showNotes').onchange = toggleNotes;
  applyExplorerSelection();
  syncCombos();
  build();
}

// Pre-tick whatever the explorer sent over. An empty or absent tsModels is the
// normal "opened with nothing selected" case, not an error - the page just
// starts blank.
function applyExplorerSelection() {
  const banner = $('handoff');
  let raw = null, sent;
  try {
    raw = localStorage.getItem('tsModels');
  } catch (e) {
    // Some browsers refuse localStorage on file:// entirely. Say so out loud
    // rather than looking like an empty selection.
    banner.textContent = 'Could not read the explorer handoff: ' + e.message
      + '. Pick combos manually below.';
    return;
  }
  if (!raw) {
    banner.textContent = 'No selection received from the explorer '
      + '(localStorage "tsModels" is empty). Pick combos below, or go back, '
      + 'lasso points in Scatter, and click Send to Time Series.';
    return;
  }
  try {
    sent = JSON.parse(raw);
  } catch (e) {
    banner.textContent = 'Explorer handoff was not valid JSON: ' + e.message;
    return;
  }
  if (!Array.isArray(sent) || !sent.length) {
    banner.textContent = 'Explorer sent an empty selection.';
    return;
  }

  // The explorer identifies a config by its grid axes; match those against the
  // combo keys baked in here. Anything it names that this page has no
  // predictions for is reported rather than silently dropped, since that means
  // the selection and the baked campaign disagree.
  const missing = [];
  let matched = 0;
  for (const m of sent) {
    const want = [m.leadtime, m.cycle, m.activation, m.num_layers, m.neurons,
                  m.dropout].map(normPart).join('|');
    const hit = DATA.order.find(label => DATA.combos[label].key === want);
    if (hit) { selected.add(hit); matched++; } else { missing.push(want); }
  }

  // Report the match explicitly. A silent zero-match is indistinguishable from
  // "nothing was sent", and that ambiguity is what made this hard to diagnose.
  if (matched && !missing.length) {
    banner.textContent = 'Received ' + sent.length + ' config(s) from the explorer; all matched.';
  } else if (matched) {
    banner.textContent = 'Received ' + sent.length + ' config(s); matched '
      + matched + ', no predictions baked for ' + missing.length + '.';
  } else {
    banner.innerHTML = 'Received ' + sent.length + ' config(s) from the explorer '
      + 'but <b>none matched</b> the combos baked into this file. '
      + 'Wanted e.g. <code>' + (missing[0] || '') + '</code>; '
      + 'this file has e.g. <code>' + (DATA.combos[DATA.order[0]] || {}).key + '</code>. '
      + 'Full lists are in the browser console (F12).';
  }
  if (missing.length) {
    console.warn('explorer sent:', sent);
    console.warn('unmatched keys:', missing);
    console.warn('keys available here:',
      DATA.order.map(l => DATA.combos[l].key));
  }
}

function normPart(v) {
  if (v === undefined || v === null || v === '') return '';
  const n = parseFloat(v);
  return isNaN(n) ? String(v).trim().toLowerCase() : String(n);
}

function syncCombos() {
  for (const box of $('combos').querySelectorAll('input')) {
    box.checked = selected.has(box.dataset.combo);
  }
}

// Select all / none across every lead-time group at once. With reps on this can
// be a lot of traces, so the MAX_TRACES guard in build() is what stops it.
function setAllLeadTimes(on) {
  for (const box of $('leadtimes').querySelectorAll('input')) box.checked = on;
  if (on) { for (const name of DATA.order) selected.add(name); }
  else { selected.clear(); }
  syncCombos();
  build();
}

init();
</script>
"""


def plot_timeseries_cove(campaign_folder, output_dir=None, datasets=None,
                         max_traces=DEFAULT_MAX_TRACES, round_to=2,
                         split_by_leadtime=None):
    """Write the time-series page(s). The entry point the runner calls.

    Returns the path written, or a LIST of paths in split mode.

    ``output_dir`` is where the rest of the suite writes its pages; the TS page
    goes there too so the explorer's "Send to Time Series" (a same-folder
    relative open) finds it.

    ``datasets=None`` means "whatever ``esb.predict`` actually produced" - the
    page bakes in every ``*_wide.csv`` it finds. That keeps this from carrying a
    second hardcoded list that silently drifts out of step with the one you
    passed to esb.predict.
    """
    campaign_folder = Path(campaign_folder)
    wide_root = campaign_folder / "predictions_wide"

    # esb.predict --split-by-leadtime leaves LT<n>/ subfolders instead of loose
    # CSVs. Detect that rather than making the caller declare it again: the
    # layout on disk is the fact, and a second flag could disagree with it.
    lt_dirs = sorted(
        (p for p in wide_root.glob("LT*") if p.is_dir()),
        key=lambda p: float(p.name[2:]))
    if split_by_leadtime is None:
        split_by_leadtime = bool(lt_dirs)
    if split_by_leadtime and not lt_dirs:
        raise FileNotFoundError(
            f"no LT<n>/ subfolders in {wide_root} - re-run esb.predict with "
            f"--split-by-leadtime to produce per-lead-time wide CSVs")

    search_dirs = lt_dirs if split_by_leadtime else [wide_root]

    if datasets is None:
        found = sorted(search_dirs[0].glob("*_wide.csv"))
        datasets = [p.name[: -len("_wide.csv")] for p in found]
        if not datasets:
            raise FileNotFoundError(
                f"no *_wide.csv in {search_dirs[0]} - "
                f"run `python -m esb.predict {campaign_folder}` first")
        print(f"datasets found: {', '.join(datasets)}")
    datasets = [str(d).strip() for d in datasets if str(d).strip()]

    # Load the campaign and the selection ONCE and reuse across every page. On a
    # 16,800-model campaign this is thousands of metadata reads; doing it per
    # page would dominate the whole bake.
    from esb.campaign import load_campaign
    by_name = {m["base_name"]: m for m in load_campaign(campaign_folder)}
    selection = load_selection(
        campaign_folder / "phase2_selection_for_phase3_tidy.csv")

    from plotly.offline import get_plotlyjs
    plotlyjs = get_plotlyjs()

    out_dir = Path(output_dir) if output_dir else campaign_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for wide_dir in search_dirs:
        suffix = f"_{wide_dir.name}" if split_by_leadtime else ""
        payload = build_payload(campaign_folder, datasets, round_to,
                                wide_dir=wide_dir, by_name=by_name,
                                selection=selection)
        print(f"{wide_dir.name}: {len(payload['order'])} combos, "
              f"lead times {payload['lead_times']}")

        html = (PAGE
                .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
                .replace("__MAX_TRACES__", str(max_traces)))

        # Inline plotly.js AFTER templating: the library source contains literal
        # braces and dollar signs that would be mangled by any templating pass.
        html = html.replace(
            "<script>\nconst DATA",
            f"<script>{plotlyjs}</script>\n<script>\nconst DATA", 1)

        out_path = out_dir / f"cove_timeseries{suffix}.html"
        out_path.write_text(html, encoding="utf-8")
        print(f"Saved: {out_path.name} "
              f"({out_path.stat().st_size / 1e6:.1f} MB)")
        written.append(str(out_path))

    if split_by_leadtime:
        # The explorer's Send-to-Time-Series opens the FIXED name
        # cove_timeseries.html (unified_plot_v17.py). In split mode no such page
        # exists, so that button would open a blank tab. Write an index under
        # that exact name instead: the button keeps working and lands somewhere
        # useful. The localStorage['tsModels'] handoff still survives, because
        # each per-lead-time page reads it on load and pre-ticks whatever combos
        # match ITS lead time.
        index = out_dir / "cove_timeseries.html"
        index.write_text(_index_page(written), encoding="utf-8")
        print(f"Saved: {index.name} (index over {len(written)} pages)")

    return written if split_by_leadtime else written[0]


def _index_page(paths) -> str:
    """A tiny link page over the per-lead-time pages. No plotly, no data."""
    rows = []
    for p in paths:
        path = Path(p)
        lt = path.stem.split("_LT")[-1]
        size_mb = path.stat().st_size / 1e6
        rows.append(
            f'<li><a href="{path.name}">Lead time {lt}h</a>'
            f'<span>{size_mb:.0f} MB</span></li>')
    return f"""<meta charset="utf-8">
<title>COVE Time Series</title>
<style>
  :root {{ --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#d8d8d8; --accent:#2a6fb5; }}
  :root[data-theme="dark"] {{ --bg:#16181c; --fg:#e8e8e8; --muted:#9aa0a6;
                              --line:#33373d; --accent:#6aa9e9; }}
  body {{ background:var(--bg); color:var(--fg); margin:0; padding:24px;
          font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }}
  h1 {{ font-size:18px; margin:0 0 4px; }}
  p {{ color:var(--muted); margin:0 0 18px; max-width:60ch; }}
  ul {{ list-style:none; padding:0; margin:0; max-width:420px; }}
  li {{ display:flex; justify-content:space-between; align-items:baseline;
        gap:12px; padding:8px 0; border-bottom:1px solid var(--line); }}
  a {{ color:var(--accent); text-decoration:none; font-weight:600; }}
  a:hover {{ text-decoration:underline; }}
  span {{ color:var(--muted); font-size:12px; white-space:nowrap; }}
</style>
<h1>COVE &mdash; Time Series</h1>
<p>One page per lead time. Each is self-contained: it carries its own data and
its own copy of plotly, so it opens straight from disk and can be shared on its
own. Traces start switched off &mdash; tick a combo to draw it.</p>
<ul>
{chr(10).join(rows)}
</ul>
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--datasets", default=None,
                    help="comma-separated datasets to bake in; default is "
                         "every *_wide.csv found in the campaign")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-traces", type=int, default=DEFAULT_MAX_TRACES)
    ap.add_argument("--round", type=int, default=2,
                    help="decimals kept per value (default 2; sensor accuracy "
                         "is ~0.1 C so more digits only cost file size)")
    split = ap.add_mutually_exclusive_group()
    split.add_argument("--split-by-leadtime", dest="split", action="store_true",
                       default=None,
                       help="one page per lead time (needs esb.predict "
                            "--split-by-leadtime). Auto-detected from the "
                            "presence of predictions_wide/LT<n>/ folders.")
    split.add_argument("--single-page", dest="split", action="store_false",
                       help="force one combined page even if LT<n>/ folders exist")
    args = ap.parse_args()

    plot_timeseries_cove(
        args.folder, output_dir=args.out_dir,
        datasets=args.datasets.split(",") if args.datasets else None,
        max_traces=args.max_traces, round_to=args.round,
        split_by_leadtime=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
