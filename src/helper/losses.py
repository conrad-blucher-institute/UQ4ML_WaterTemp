"""
This file is for the use of the Coastal Dynamics Laboratory

Custom loss functions for training machine learning models.

Refactored out of src/helper/utils.py
"""

# Continuous Rank Probability Score Loss Function.
# CRPS is defined once as the evaluation metric in metrics.py; re-exported here
# under the crps_loss name so the training loss and metric never diverge.
try:
    from src.helper.metrics import crps as crps_loss
except ImportError:
    from metrics import crps as crps_loss



#experimental custom loss function
def weighted_mse(CS_weight):
    # Importing libraries
    import keras.backend as K

    def loss(target, prediction):
        min_temp = K.minimum(K.abs(target), K.abs(prediction))

        # Shape is really appealing, but doesn't seem to work.
        # CS_weight can't be negative
        #se_weight = 1/(min_temp)**CS_weight

        # Works for CS_weight in [0, 5]
        # 0 is MSE
        se_weight = (40 - min_temp)**CS_weight

        se = K.abs(prediction - target) ** 2
        weighted_se = se_weight * se
        weighted_mse = K.mean(weighted_se)
        return weighted_mse
    return loss
