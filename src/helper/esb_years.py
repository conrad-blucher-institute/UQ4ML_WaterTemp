"""Which ESB year CSV plays which role. No third-party imports, by design.

These live apart from utils_mse_crps so that code needing only the year names --
`esb.retrain_final --dry-run`, provenance checks, packaging -- can read them
without importing TensorFlow. utils_mse_crps re-exports them, so every existing
`from src.helper.utils_mse_crps import ESB_ROTATION_YEARS` keeps working and
there is still exactly ONE definition.
"""

# ORDER-SENSITIVE: this is year2..year5, and splittingData rotates the list to
# derive train/val/test per cycle. Reordering changes every cycle's split.
ESB_ROTATION_YEARS = (
    "esb_2021_2022",
    "esb_2022_2023",
    "esb_2023_2024",
    "esb_2024_2025",
)

# Held out of the rotation entirely. Winter Storm Uri.
ESB_INDEPENDENT_YEARS = ("esb_2020_2021",)
