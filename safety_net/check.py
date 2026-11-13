"""Safety net for the ESB refactor ("same inputs -> same outputs").

    python safety_net/check.py                  # run all 3 checks (~30 s)
    python safety_net/check.py --skip-smoke     # skip the slow training run
    python safety_net/check.py --freeze         # re-save digest + viz numbers (own commit!)
    python safety_net/check.py --freeze-metrics 20   # re-measure the metric band from 20 runs

Checks: (a) training-input digest exact, (b) mape_smoke metrics inside baseline.json band,
(c) numbers behind the run_v17 plots exact. Exit code is non-zero if any check FAILs.
"""
import base64, contextlib, io, json, os, re, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VIZ = ROOT / "esb_03_2026_tune_experiment" / "tuning-viz-folder"
BASE = HERE / "baseline.json"
COLS = ["val_mae", "val_mae12", "val_me", "val_me12", "val_mape",
        "mae_2021", "mae12_2021", "me_2021", "me12_2021", "mape_2021"]
dec = json.JSONDecoder()


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def digest():
    from esb import _verify
    return quiet(_verify.compute_fit_inputs, _verify._smoke_config())["digest"]


def smoke_row():  # one real mape_smoke run into a fresh temp dir (no resume, nothing left behind)
    from esb.cli import main
    with tempfile.TemporaryDirectory() as out:
        quiet(main, ["run", "--profile", "mape_smoke", "--output_dir", out])
        return pd.read_csv(Path(out) / "mape_progress.csv").iloc[0]


def band_from(runs):  # runs: DataFrame, one row per smoke run. Band = [min - 0.5*range, max + 0.5*range]
    out = {}
    for c in COLS:
        lo, hi = runs[c].min(), runs[c].max()
        floor = 0.0 if ("mae" in c or "mape" in c) else -1e99  # MAE/MAPE cannot be negative
        out[c] = dict(median=round(runs[c].median(), 4), min=round(lo, 4), max=round(hi, 4),
                      lo=round(max(lo - 0.5 * (hi - lo), floor), 4), hi=round(hi + 0.5 * (hi - lo), 4))
    return out


def flat(x, key=""):  # (key, repr(value)) for every leaf; style keys (colors, fonts...) are skipped
    if isinstance(x, dict) and "bdata" in x:  # newer plotly packs arrays as base64: unpack so plotly version doesn't matter
        x = np.frombuffer(base64.b64decode(x["bdata"]), x["dtype"]).tolist()
    if isinstance(x, dict):
        for k, v in x.items():
            if not re.search(r"color|font|template|axis|showscale|^type$", k):
                yield from flat(v, f"{key}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, f"{key}[{i}]")
    else:
        yield key, repr(x)


def page_numbers(s):  # the data baked into one plot page: Plotly traces, JS data vars, HTML tables
    rows = []
    for i, m in enumerate(re.finditer(r"Plotly\.newPlot\(\s*[\"'][^\"']*[\"']\s*,\s*(?=\[)", s)):
        rows += flat(dec.raw_decode(s, m.end())[0], f"plot{i}")
    for m in re.finditer(r"^\s*(?:var|const|let) (allData|allRecords|filterValues) = ", s, re.M):
        rows += flat(dec.raw_decode(s, m.end())[0], m.group(1))
    rows += [(f"td{i}", t.strip()) for i, t in enumerate(re.findall(r"<td>(.*?)</td>", s, re.S))]
    return rows


def viz_numbers():  # run run_v17 on the frozen smoke CSV; return {page: [(key, value)]}
    sys.path.insert(0, str(VIZ))
    import run_v17 as rv
    with tempfile.TemporaryDirectory() as t:
        shutil.copy(HERE / "frozen_smoke" / "mape_progress.csv", t)
        data = rv.load_data(rv.discover_csvs(Path(t)), "val_mae")
        quiet(rv.run_v17, data, "val_mae", str(Path(t) / "_visuals"), results_folder=t)
        out = {"loaded_data": list(flat(data.to_dict("records")))}
        for f in sorted((Path(t) / "_visuals").glob("*.html")):
            out[f.stem] = page_numbers(f.read_text(encoding="utf-8"))
    return out


def main():
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)  # the profile and data paths are relative to the repo root
    base = json.loads(BASE.read_text())
    args = sys.argv[1:]

    if "--freeze" in args:
        base["digest"] = digest()
        BASE.write_text(json.dumps(base, indent=2))
        (HERE / "viz_numbers").mkdir(exist_ok=True)
        for name, rows in viz_numbers().items():
            pd.DataFrame(rows, columns=["key", "value"]).to_csv(HERE / "viz_numbers" / f"{name}.csv", index=False)
        return print("frozen: digest", base["digest"][:12], "+ viz_numbers/*.csv")
    if "--freeze-metrics" in args:
        n = int(args[args.index("--freeze-metrics") + 1])
        base["n_runs"], base["metrics"] = n, band_from(pd.DataFrame([smoke_row() for _ in range(n)]))
        BASE.write_text(json.dumps(base, indent=2))
        return print("frozen: metric band from", n, "runs")

    res = []  # (check name, ok, detail); ok=None means skipped
    d = digest()
    res.append(("input fingerprint", d == base["digest"], d[:12] if d == base["digest"] else f"got {d[:12]}, want {base['digest'][:12]}"))

    if "--skip-smoke" in args:
        res.append(("smoke metrics", None, "skipped (--skip-smoke)"))
    else:
        row = smoke_row()
        bad = [] if row["status"] == "completed" else [f"status={row['status']}"]
        for c, b in base["metrics"].items():
            v = float(row[c]) if row["status"] == "completed" else float("nan")
            if not b["lo"] <= v <= b["hi"]:  # NaN fails this test too
                bad.append(f"{c}={v:.4g} outside [{b['lo']}, {b['hi']}]")
        res.append(("smoke metrics", not bad, "; ".join(bad) or f"val_mape={row['val_mape']:.4g}, val_mae={row['val_mae']:.4g}, all {len(COLS)} metrics in band"))

    got, bad = viz_numbers(), []
    for f in sorted(set(got) | {p.stem for p in (HERE / "viz_numbers").glob("*.csv")}):
        p = HERE / "viz_numbers" / f"{f}.csv"
        want = list(pd.read_csv(p, dtype=str, keep_default_na=False).itertuples(index=False, name=None)) if p.exists() else None
        if want != got.get(f):
            bad.append(f"{f} differs")
    res.append(("viz numbers", not bad, "; ".join(bad) or f"{len(got)} pages identical"))

    for name, ok, detail in res:
        print(f"{'SKIP' if ok is None else 'PASS' if ok else 'FAIL'}  {name:18s} {detail}")
    sys.exit(1 if any(ok is False for _, ok, _ in res) else 0)


if __name__ == "__main__":
    main()
