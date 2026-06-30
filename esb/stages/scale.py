"""scale stage (façade) — NO-OP passthrough in Stage B.

The current ESB pipeline has no scaler (CV-leakage verdict: nothing to leak).
This stage is the seam where Stage C reconciles the two StandardScaler impls
(esb_dev_normalization + the inference path). Until then it is an identity
passthrough so the strangler-fig baseline is a true no-op on this axis.
"""


def scale(arrays):
    """Identity passthrough (no scaler exists yet — Stage C)."""
    return arrays
