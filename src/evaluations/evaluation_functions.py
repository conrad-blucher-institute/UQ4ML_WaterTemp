#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:06:55 2025

@author: Jarett Woodall and the Cool Turtle Team

****NOTE: The functions get_pit_points() and get_spread_skill_points()
were created by Dr. Ryan Lagerquist and taken from this GitHub Repository:
https://github.com/thunderhoser/cira_uq4ml
    
"""

##### IMPORTS #####
import tensorflow as tf 

from tensorflow.python.ops.numpy_ops import np_config 

import numpy as np

import scipy

import scipy.stats

import tensorflow_probability as tfp

np_config.enable_numpy_behavior()

######## Performance Evaluation Functions ########

def mae(y_true, y_pred):
    """
    Compute the Mean Absolute Error (MAE).
    
    Parameters:
    y_true : tf.Tensor
    y_pred : tf.Tensor

    Returns:
    float : The MAE score.
    """
    if y_pred.shape[1] > 1:
        mean_pred = tf.reduce_mean(y_pred, axis=-1)
        mean_pred = tf.expand_dims(mean_pred, axis=-1)
    else:
        mean_pred = y_pred

    differences = tf.abs(tf.subtract(y_true, mean_pred))
    score = tf.reduce_mean(differences)

    return score.numpy()

# END: def mae()


def mae12(y_true, y_pred):
    """
    Compute the MAE for targets < 12.
    
    Parameters:
    y_true : tf.Tensor
    y_pred : tf.Tensor

    Returns:
    float
        The MAE score for values < 12.
    """
    if y_pred.shape[1] > 1:
        mean_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)
        true_val = y_true[:, :1]
    else:
        mean_pred = y_pred
        true_val = y_true

    mask = true_val < 12
    filtered_y_true = tf.boolean_mask(true_val, mask)
    filtered_y_pred_mean = tf.boolean_mask(mean_pred, mask)

    differences = tf.abs(filtered_y_true - filtered_y_pred_mean)
    score = tf.reduce_mean(differences).numpy()

    return score

# END: def mae12()


def me(y_true, y_pred):
    
    """
    Compute Mean Error (ME), or bias.
    
    Parameters:
    y_true : tf.Tensor
    y_pred : tf.Tensor
    
    Returns:
    float
        ME score.
    """

    if y_pred.shape[1] > 1:
        mean_pred = tf.reduce_mean(y_pred, axis=-1)
        mean_pred = tf.expand_dims(mean_pred, axis=-1)
    else:
        mean_pred = y_pred

    mean_square = tf.reduce_mean((tf.subtract(y_true, mean_pred)), axis=-1)
    score = tf.reduce_mean(mean_square)

    return score.numpy()

# END: def me()


def me12(y_true, y_pred):
    """
    Compute the Mean Error (ME) for predictions where the true value < 12.
    
    Parameters:
    y_true : tf.Tensor
    y_pred : tf.Tensor

    Returns:
    float
        The ME score for values < 12.
    """
    if y_pred.shape[1] > 1:
        mean_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)
        true_val = y_true[:, :1]
    else:
        mean_pred = y_pred
        true_val = y_true

    mask = true_val < 12
    filtered_y_true = tf.boolean_mask(true_val, mask)
    filtered_y_pred_mean = tf.boolean_mask(mean_pred, mask)

    differences = filtered_y_true - filtered_y_pred_mean
    score = tf.reduce_mean(differences).numpy()

    return score

# END: def me12()


def mse(y_true, y_pred):
    """
    Compute Mean Squared Error (MSE).
    
    Parameters:
    y_true : tf.Tensor
    y_pred : tf.Tensor

    Returns:
    float
        MSE score.
    """
    if y_pred.shape[1] > 1:
        mean_pred = tf.reduce_mean(y_pred, axis=-1)
        mean_pred = tf.expand_dims(mean_pred, axis=-1)
    else:
        mean_pred = y_pred

    mean_square = tf.reduce_mean(tf.square(tf.subtract(y_true, mean_pred)), axis=-1)
    score = tf.reduce_mean(mean_square)

    return score.numpy()

# END: def mse()


def rmse_avg(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE) averaged across the batch.
    
    Parameters:
    y_true : tf.Tensor
        Ground truth values.
    y_pred : tf.Tensor
        Predicted values.

    Returns:
    float
        The RMSE score.
    """
    mean_pred = y_pred
    root_mean_square = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, mean_pred))))
    score = tf.reduce_mean(root_mean_square)

    return score.numpy()

# END: def rmse_avg()

######## Uncertainty Quantification Evaluation Functions ########

def ssrat_avg(y_true, y_pred, y_pred_std):
    """
    Compute the average Spread-Skill Ratio (SSRAT).
    
    Parameters:
    y_true : tf.Tensor
        True values.
    y_pred : tf.Tensor
        Predicted values.
    y_pred_std : tf.Tensor
        Standard deviation of the predictions.

    Returns:
    float
        The SSRAT score.
    """
    ssrat_score = tf.math.reduce_mean(y_pred_std) / rmse_avg(y_true, y_pred)

    return ssrat_score.numpy()

# END: def ssrat_avg()


@tf.function # Enables graph execution
def crps_gaussian_tf(mu, sigma, y_true):
    """
    Compute the CRPS for a Gaussian predictive distribution in TensorFlow.
    
    Parameters
    ----------
    mu : tf.Tensor
        Predicted mean, shape [batch_size, ...]
    sigma : tf.Tensor
        Predicted std dev, shape [batch_size, ...]
    y_true : tf.Tensor
        True values, same shape as mu
    """
    sigma = tf.clip_by_value(sigma, 1e-8, 1e8)
    z = (y_true - mu) / sigma

    normal = tfp.distributions.Normal(loc=0.0, scale=1.0)
    pdf_z = normal.prob(z)
    cdf_z = normal.cdf(z)

    two_cdf_z_minus_1 = 2.0 * cdf_z - 1.0
    term1 = z * two_cdf_z_minus_1
    term2 = 2.0 * pdf_z
    term3 = 1.0 / tf.sqrt(tf.constant(np.pi, dtype=z.dtype))

    crps_per_sample = sigma * (term1 + term2 - term3)

    return tf.reduce_mean(crps_per_sample)

# END: def crps_gaussian_tf()


def get_pit_points(y_true, y_pred, y_std):
    """
    Compute PITD (Probability Integral Transform Distance) score for calibrated predictions.
    
    Parameters:
    y_true : np.ndarray
    y_pred : np.ndarray
    y_std : np.ndarray

    Returns:
    float
        PITD score.
        
    Author: Ryan Lagerquist
    Source: https://github.com/thunderhoser/cira_uq4ml
    """
    pit_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def get_pit_dvalue(pit_counts):
        dvalue = 0.
        nbins = pit_counts.shape[0]
        nbinsI = 1. / nbins

        pitTot = np.sum(pit_counts)
        pit_freq = np.divide(pit_counts, pitTot)
        for i in range(nbins):
            dvalue += (pit_freq[i] - nbinsI) ** 2
        dvalue = np.sqrt(dvalue / nbins)
        return dvalue

    def get_histogram(var, bins=10, density=False, weights=None):
        counts, bin_edges = np.histogram(
            var, bins=bins, density=density, weights=weights)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return counts, bin_centers

    nSamples = y_true.size
    pit_values = scipy.stats.norm.cdf(x=y_true, loc=y_pred, scale=y_std).reshape(-1)
    weights = np.ones_like(pit_values) / nSamples
    pit_counts, bin_centers = get_histogram(pit_values, bins=pit_bins, weights=weights)

    pitd_score = get_pit_dvalue(pit_counts)

    return pitd_score

# END: def get_pit_points()


def get_spread_skill_points(y_true, y_pred, y_std, nBins=10, bins=None, showR2=True, spread_last=None, verbose=True):
    """
    Calculate spread-skill relationship as a diagnostic for ensemble reliability.
    
    Parameters:
    y_true : np.ndarray
    y_pred : np.ndarray
    y_std : np.ndarray
    nBins : int
    bins : list or None
    showR2 : bool
    spread_last : any
    verbose : bool

    Returns:
    float
        Spread-skill reliability score (lower is better).
        
    Author: Ryan Lagerquist
    Source: https://github.com/thunderhoser/cira_uq4ml
    """
    def create_contours(minVal, maxVal, nContours, match=False):
        if match:
            xVal = np.max([np.abs(minVal), np.abs(maxVal)])
            interval = 2 * xVal / (nContours - 1)
        else:
            interval = (maxVal - minVal) / (nContours - 1)

        contours = np.empty((nContours))
        for i in range(nContours):
            contours[i] = minVal + i * interval
        return contours

    if y_true.shape != y_pred.shape:
        print("Mismatching shapes:")
        print(f"   y_true: {y_true.shape}")
        print(f"   y_pred: {y_pred.shape}")
        return {}

    nPts = y_true.size

    if not bins:
        minBin = np.min([0., y_std.min()])
        maxBin = np.ceil(np.max([rmse_avg(y_true, y_pred), y_std.max()]))
        bins = create_contours(minBin, maxBin, nBins + 1)
    else:
        nBins = len(bins) - 1

    ssRel = 0.
    error = np.zeros((nBins)) - 999.
    spread = np.zeros((nBins)) - 999.
    y_on_error = np.zeros((y_pred.shape)) - 999.

    for i in range(nBins):
        refs = np.logical_and(y_std >= bins[i], y_std < bins[i + 1])
        nPtsBin = np.count_nonzero(refs)
        if nPtsBin > 0:
            ytrueBin = y_true[refs]
            ymeanBin = y_pred[refs]
            error[i] = rmse_avg(ytrueBin, ymeanBin)
            spread[i] = np.mean(y_std[refs])
            y_on_error[refs] = np.abs(y_true[refs] - y_pred[refs])
            ssRel += (nPtsBin / nPts) * np.abs(error[i] - spread[i])

    return ssRel

# END: def get_spread_skill_points()
