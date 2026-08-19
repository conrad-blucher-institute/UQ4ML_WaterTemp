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


def feature_names(input_structure, input_hours_forecast, atp_hours_back,
                  wtp_hours_back, pred_atp_interval) -> list[str]:
    """The model's input columns, IN ORDER — the contract a consumer must match.

    This is the one part of the input contract that cannot live inside the
    ``.keras``: the frozen Normalization layer applies mean/scale positionally
    and physically cannot notice a reordered feature matrix. A wrong order is
    therefore silently wrong (same width, plausible numbers, bad forecasts), so
    the order ships beside the model as data and consumers assert against it.

    Derived from the same four knobs that drive :func:`features`, so the width
    is ``wtp_hours_back + 1 + atp_hours_back + 1 + input_hours_forecast /
    pred_atp_interval`` by construction rather than by measurement.

    Only ``"descending"`` is implemented — that is what every scaled campaign
    used. ``"ascending"`` raises rather than guessing: emitting a plausible but
    wrong order would defeat the entire point of this function.
    """
    if input_structure != "descending":
        raise NotImplementedError(
            f"feature_names() only knows the 'descending' column order; got "
            f"{input_structure!r}. Refusing to guess — a wrong order here is "
            f"silently wrong downstream."
        )

    w = int(wtp_hours_back)
    a = int(atp_hours_back)
    f = int(input_hours_forecast)
    i = int(pred_atp_interval)
    if i <= 0 or f % i:
        raise ValueError(
            f"input_hours_forecast ({f}) must be a positive multiple of "
            f"pred_atp_interval ({i})"
        )

    names: list[str] = []
    # Backward blocks are REVERSED: largest lag first.
    names += [f"waterTemperature__{h}h_ago" for h in range(w, 0, -1)]
    names.append("Water Average")
    names += [f"airTemperature__{h}h_ago" for h in range(a, 0, -1)]
    names.append("Air Average")
    # Forecast block ascends by forecast hour.
    names += [f"airTemperature_pred__{h}h_forecast" for h in range(i, f + 1, i)]
    return names
