# Safety net (card #10)

A "before" snapshot so every later change can be checked: **same inputs, same outputs**.

    python safety_net/check.py               # all 3 checks, ~30 s, prints PASS/FAIL, exit code 1 on any FAIL
    python safety_net/check.py --skip-smoke  # skip the training run (checks a and c only, ~5 s)

Run it from any folder inside the `uq4ml` environment (TF 2.15). Locally or in a Slurm job, same command.

## What it checks

| # | Check | How | Strictness |
|---|-------|-----|-----------|
| a | Input fingerprint | `esb/_verify.py` digest of the `model.fit` inputs for `mape_smoke` vs the Stage B digest `d80ae54...` (full value in `baseline.json`) | exact |
| b | Smoke metrics | real `python -m esb run --profile mape_smoke` (1 config, 2 epochs) into a temp dir; 10 metrics must be finite and inside the band in `baseline.json` | tolerance |
| c | Viz numbers | `run_v17.py` functions run on a **frozen** smoke CSV (`frozen_smoke/mape_progress.csv`); the numbers baked into each plot page are compared to `viz_numbers/*.csv` | exact |

For (c) the HTML itself is never stored (the repo guard blocks `*.html`); only the numbers behind the plots are
(Plotly traces, embedded data variables, table cells). Colors, fonts and layout are ignored, so restyling a plot does not trip it.
Needs `tkinter` importable (`run_v17.py` imports it), no display.

## Why the metric tolerance is a wide band

TF weight init is unseeded, so metrics differ on every run. Measured over **20 real smoke runs** (2 epochs):

| metric | min | median | max | band used |
|---|---|---|---|---|
| val_mape | 8.9861 | 62.3715 | 255.9143 | [0.0, 379.3783] |
| val_mae | 1.8132 | 15.2633 | 61.952 | [0.0, 92.0215] |
| mape_2021 | 9.8545 | 62.5452 | 256.3737 | [0.0, 379.6332] |

(all 10 metrics are in `baseline.json`). After 2 epochs the result is dominated by the random start, so the spread is ~28x
(val_mape 9 to 256) and a tight relative tolerance would fail healthy runs. A standard deviation is also misleading here (heavy tail).
So the band is the observed envelope widened by half its range each side: `[min - 0.5*range, max + 0.5*range]`, MAE/MAPE floored at 0.
Leave-one-out test: building the band from 19 runs and testing the 20th, 0 of 20 runs fell outside.

**This check is a tripwire, not a precision test.** It catches crashes, NaN/inf, a run that did not train, or a blow-up.
It will NOT catch a small accuracy regression. For that, TF would need a fixed seed (a code change, so out of scope for this card).

## Updating the baseline

Baselines change only on purpose, and **always in their own commit** (no code changes in the same commit), with the reason in the message.

    python safety_net/check.py --freeze            # re-save digest + viz_numbers/ (after an intended data-prep or viz change)
    python safety_net/check.py --freeze-metrics 20 # re-measure the metric band from 20 smoke runs (~5 min)

To change the frozen viz input, copy a new `mape_progress.csv` from a smoke run into `frozen_smoke/` (keep it tiny), then `--freeze`.
Never put these files under a folder named `results/` (gitignored by the guard).
