"""model stage (façade) — delegates to the grid-search tuner's _build_model.

Owns: architecture + loss/metric wiring. No logic moved (Stage B): the live
path builds the model inside the tuner. This façade exposes the same builder
for Stage C/D when model construction migrates out of the tuner.
"""


def build_model(tuner, config):
    """Build a keras model. Façade over BaseHyperparameterTuner._build_model()."""
    return tuner._build_model(config)
