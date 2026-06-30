"""train stage (façade) — delegates to the grid-search tuner's _train_model.

Owns: callbacks, batch size, epochs, fit loop. No logic moved (Stage B): the
live path trains inside the tuner worker. This façade names the seam for when
training migrates out of the tuner in Stage D.
"""


def train_model(tuner, model, config):
    """Train one config. Façade over BaseHyperparameterTuner._train_model()."""
    return tuner._train_model(model, config)
