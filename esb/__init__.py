"""esb — single front door for the ESB water-temperature experiment pipeline.

Stage B (strangler-fig skeleton): a thin shell over the EXISTING grid-search
MAPE/MSE tuner. The Config schema (config.py) is the single source of truth that
generates the CLI and is introspectable for a future GUI. No pipeline logic has
moved yet — stages/ are thin façades over src/helper/utils_mse_crps.py and the
orchestration delegates to the existing tuner's run_tuning().

See docs/ESB_PIPELINE_DESIGN.md and docs/ESB_STAGE_B_REPORT.md.
"""

__all__ = ["config", "cli", "pipeline"]
