# ESB Pipeline — Current State (as-is)

> Approximate snapshot of how things work **today**, for contrast with
> `ESB_PIPELINE_DESIGN.md` (the target). Sketched 2026-06-29, not audited — "more or less."

## 1. Entry points today (fragmented — no single front door)

```mermaid
flowchart TB
  user(["user must remember<br/>which script + which flags"])

  user --> A["crps_mme_runner.py<br/>+ my_parser.py<br/>+ configs/tuner_*.txt"]
  user --> B["run_all_tuners.py<br/>→ mse_tuner / mape_tuner<br/>→ base_tuner → tuner_utils"]
  user --> C["operational_mse_crps_driver.py<br/>hardcoded hyperparams<br/>(big if/elif tree)"]
  user --> D["pnn_mme_driver.py<br/>(old utils_pnn, 10-yr)"]

  A --> P["preparingData()<br/>(utils_mse_crps.py)"]
  B --> P
  C --> P
  D --> P2["utils_pnn pipeline<br/>(separate, diverged)"]

  P --> K["keras / keras_tuner<br/>build + fit"]
  C --> K
  K --> R[("results: predictions.csv<br/>*.keras · *.pkl")]
```

**Pain:** four-ish ways to launch, each with its own flag/config surface. You carry the
mapping in your head (which script, which flags, which config file).

## 2. What happens inside `preparingData()` (one big function)

```mermaid
flowchart TB
  csv[("ESB CSVs<br/>sorted glob;<br/>independent year skipped")] --> rd["readingData()<br/>list of DataFrames"]
  rd --> ca["creatingAdditionalColumns()<br/>vectorized shift; input_structure"]
  ca --> sp["splittingData()<br/>4-year rotation by cycle"]
  sp --> cm["countingMissingValues()<br/>deletingMissingValues()<br/>(-999 + NaN)"]
  cm --> dc["dataframe_checker()<br/>(note: calls exit() on failure)"]
  dc --> rs["reshaping()<br/>col_start = 1 desc / 3 asc"]
  rs --> arr["x/y train,val,test<br/>+ dates + air temps"]
  arr --> ret(["return tuple → caller"])

  note["NO scaling/normalization stage<br/>(raw values straight to model)"]:::warn
  arr -.-> note

  classDef warn fill:#ffecec,stroke:#d33;
```

**Pain:** all stages live in one function with debug prints/CSV dumps interleaved; the
`col_start` rule is duplicated wherever arrays are sliced; there is **no scale stage**, so
models extrapolate badly on out-of-distribution inputs (Uri 2021).

## 3. Inference path (separate, bolted on)

```mermaid
flowchart LR
  indep[("independent year CSV<br/>e.g. esb_2020_2021.csv")] --> pi["prepare_independent_year()<br/>re-runs feature engineering<br/>on a single CSV"]
  pi --> ev["model.evaluate / predict"]
  note2["assumes csv_files[0] is the<br/>independent year; no shared<br/>scaler with training"]:::warn
  pi -.-> note2
  classDef warn fill:#ffecec,stroke:#d33;
```

## 4. Known warts to fix in the redesign (from CLAUDE.md / this session)
- **`crps_loss()` is actually MAE** — "CRPS" models trained through this pipeline are MAE-trained.
- **Double `model.fit()`** in `operational_mse_crps_driver.py` (trains twice per iteration).
- **Duplicate variable defs** (inputShape/batch_size, callbacks) in the operational driver.
- **`exit()` inside `dataframe_checker()`** — kills the process instead of raising.
- **No normalization** anywhere in the ESB path.
- **Scattered entry points** with overlapping-but-different flag surfaces.
- **`col_start` invariant** (1 desc / 3 asc) repeated instead of owned in one place.

---

### How this maps to the target (`ESB_PIPELINE_DESIGN.md`)
| Today | Target |
|-------|--------|
| 4 entry scripts, flags in your head | 1 command + named profile + defaults |
| `preparingData()` monolith | cohesive `read → features → clean → split → reshape` stages |
| no scaling | dedicated `scale` stage (reconciled, fit-on-train-only) |
| inference bolted on | `infer` stage sharing the fitted scaler + `column_map` |
| `exit()` / silent issues | loud validation + inter-stage asserts |
