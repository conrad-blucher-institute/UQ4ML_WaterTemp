"""
This file is for the use of the Coastal Dynamics Laboratory

Custom loss functions for training machine learning models.

Refactored out of src/helper/utils.py
"""

# Continuous Rank Probability Score Loss Function
def crps_loss(y_true, y_pred):
    """
    From Ryan Lagerquist...

    Calculates the Continuous Ranked Probability Score (CRPS)
    for finite ensemble members and a single target.

    This implementation is based on the identity:
        CRPS(F, x) = E_F|y_pred - y_true| - 1/2 * E_F|y_pred - y_pred'|
    where y_pred and y_pred' denote independent random variables drawn from
    the predicted distribution F, and E_F denotes the expectation
    value under F.

    Following the approach by Steven Brey at
    TheClimateCorporation (formerly ClimateLLC)
    https://github.com/TheClimateCorporation/properscoring

    Adapted from David Blei's lab at Columbia University
    http://www.cs.columbia.edu/~blei/ and
    https://github.com/blei-lab/edward/pull/922/files


    References
    ---------
    Tilmann Gneiting and Adrian E. Raftery (2005).
        Strictly proper scoring rules, prediction, and estimation.
        University of Washington Department of Statistics Technical
        Report no. 463R.
        https://www.stat.washington.edu/research/reports/2004/tr463R.pdf

    H. Hersbach (2000).
        Decomposition of the Continuous Ranked Probability Score
        for Ensemble Prediction Systems.
        https://doi.org/10.1175/1520-0434(2000)015%3C0559:DOTCRP%3E2.0.CO;2
    """

    import tensorflow as tf

    # Variable names below reference equation terms in docstring above
    term_one = tf.reduce_mean(tf.abs(
        tf.subtract(y_true, y_pred)), axis=-1)

    term_two = tf.reduce_mean(
        tf.abs(
            tf.subtract(tf.expand_dims(y_pred, -1),
                        tf.expand_dims(y_pred, -2))),
        axis=(-2, -1))

    half = tf.constant(-0.5, dtype=term_two.dtype)

    score = tf.add(term_one, tf.multiply(half, term_two))

    score = tf.reduce_mean(score)

    return score


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
