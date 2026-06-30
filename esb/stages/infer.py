"""infer stage (façade) — delegates to utils_mse_crps.prepare_independent_year.

Owns: independent-year feature engineering for evaluation (e.g. 2021). No logic
moved (Stage B): the live path calls prepare_independent_year inside the tuner's
post-hoc 2021 evaluation. Stage C will add the shared-scaler + column_map here.
"""
from esb.stages import _fn


def prepare_independent_year(csv_path, input_structure, lead_time,
                             atp_hours_back, wtp_hours_back,
                             pred_atp_interval=1, ipp_offset=0.0):
    """Prepare an independent test year. Façade over prepare_independent_year()."""
    return _fn("prepare_independent_year")(
        csv_path, input_structure, lead_time, atp_hours_back, wtp_hours_back,
        pred_atp_interval, ipp_offset,
    )
