# Archive — historical records, do not update

These documents describe what was true on a given date. They are **append-only
history**, not living documentation: they hold the frozen golden digests and the
locked decisions that existing campaign results are justified by. If someone asks
"why should I trust `results/esb_tuner_scaled`?", the answer is in here.

Do not edit them to reflect current behavior. When behavior changes, record the
change in the living docs instead:

| Living doc | Covers |
|---|---|
| `../ESB_PIPELINE_DESIGN.md` | where the pipeline is going + locked decisions (C1-C4) |
| `../ESB_PIPELINE_CURRENT.md` | how it works right now |

## Contents

| File | What it records |
|---|---|
| `ESB_STAGE_A_REPORT.md` | Stage A audit + decisions |
| `ESB_STAGE_B_REPORT.md` | Stage B skeleton; unscaled golden digest `d80ae54…` |
| `ESB_STAGE_C_REPORT.md` | Stage C scale stage as originally built (pre-C4: sidecar `.joblib`, `ScaledArrays`) |
| `ESB_STAGE_E_REPORT.md` | Stage E GUI + `--shard` sharding |
| `ESB_REFACTOR_STAGE_PROMPTS.md` | The A→B→C→E→D→F execution plan. **Stages D and F are still outstanding** — that plan has not been superseded, only the paste-ready prompt format has. |

> **Superseded by decision C4 (2026-08-14):** the Stage C report describes a
> `StandardScaler` persisted as `_scaler.joblib` beside each model, and an
> `Arrays → ScaledArrays` contract. Both are gone. The scaler is now a frozen
> `keras.layers.Normalization` layer inside the `.keras`, and the column-order
> contract ships as `<base>_metadata.json`. See DESIGN §6, decision C4.
