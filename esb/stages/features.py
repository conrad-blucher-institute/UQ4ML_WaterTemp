"""features stage (façade) — delegates to utils_mse_crps.creatingAdditionalColumns.

Owns: vectorized shift, input_structure ordering. No logic moved (Stage B).
"""
from esb.stages import _fn


def features(df, input_structure, input_hours_forecast, atp_hours_back,
             wtp_hours_back, pred_atp_interval, ipp_offset=0.0):
    """Engineer lag/forecast columns. Façade over creatingAdditionalColumns()."""
    return _fn("creatingAdditionalColumns")(
        df, input_structure, input_hours_forecast, atp_hours_back,
        wtp_hours_back, pred_atp_interval, ipp_offset,
    )
