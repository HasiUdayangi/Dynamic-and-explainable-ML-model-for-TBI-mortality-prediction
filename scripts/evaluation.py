#!/usr/bin/env python3
"""
Evaluation Module for Ectopic Beat Detection Model

This module provides functions for:
  - Computing custom evaluation metrics (precision, recall, Matthews correlation coefficient)
  - Bootstrapping evaluation metrics (AUC, Accuracy, Precision, Recall, Matthews)
  - Calculating hourly metrics (AUC and PRC AUC)
  - A custom Keras Callback (MetricsHistory) to record evaluation metrics on the validation set per epoch
  - Calibration functions: computing calibration curves and visualizing calibration plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
from sklearn.utils import resample
from sklearn.calibration import calibration_curve, CalibrationDisplay

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

import tensorflow as tf

# -----------------------------
# Custom Metrics using Keras backend
# -----------------------------
def precision(y_true, y_pred):
    """Calculate precision using Keras backend."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    """Calculate recall using Keras backend."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def matthews(y_true, y_pred):
    """Calculate Matthews correlation coefficient using Keras backend."""
    y_pred_pos = K.cast(K.round(K.clip(y_pred, 0, 1)), 'float32')
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.cast(K.round(K.clip(y_true, 0, 1)), 'float32')
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * (1 - y_pred_pos))

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())

def np_matthews(y_true, y_pred):
    """Calculate Matthews correlation coefficient using NumPy."""
    y_pred_pos = np.round(y_pred).astype(int)
    y_neg = 1 - y_true
    y_pos = y_true
    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * (1 - y_pred_pos))
    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * (1 - y_pred_pos))
    numerator = (tp * tn - fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + np.finfo(float).eps)

# -----------------------------
# Bootstrapping Metrics
# -----------------------------
def bootstrap_metrics(y_true, y_pred_prob, n_bootstraps=1000):
    """
    Bootstrap evaluation metrics for AUC, Accuracy, Precision, Recall, and Matthews.
    
    Parameters:
      y_true: Ground truth binary labels (numpy array).
      y_pred_prob: Predicted probabilities (numpy array).
      n_bootstraps: Number of bootstrap iterations.
    
    Returns:
      A dictionary containing lists of bootstrapped metrics.
    """
    y_pred = (y_pred_prob >= 0.5).astype(int)
    boot_metrics = {key: [] for key in ['AUC', 'Accuracy', 'Precision', 'Recall', 'Matthews']}
    for _ in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)), replace=True)
        bs_y_true = y_true[indices]
        bs_y_pred_prob = y_pred_prob[indices]
        bs_y_pred = y_pred[indices]
        boot_metrics['AUC'].append(roc_auc_score(bs_y_true, bs_y_pred_prob))
        boot_metrics['Accuracy'].append(accuracy_score(bs_y_true, bs_y_pred))
        boot_metrics['Precision'].append(precision_score(bs_y_true, bs_y_pred, zero_division=0))
        boot_metrics['Recall'].append(recall_score(bs_y_true, bs_y_pred, zero_division=0))
        boot_metrics['Matthews'].append(np_matthews(bs_y_true, bs_y_pred))
    return boot_metrics

# -----------------------------
# Hourly Metrics Calculation
# -----------------------------
def calculate_hourly_metrics(y_true, y_pred):
    """
    Calculates hourly AUC and Precision-Recall AUC for hourly aggregated data.
    Expects y_true and y_pred to be lists or arrays of arrays (one per hour).
    
    Returns:
      Two lists: hourly AUC and hourly PRC AUC.
    """
    n_hours = len(y_true)
    hourly_auc = []
    hourly_prc_auc = []
    for h in range(n_hours):
        y_true_hour = y_true[h]
        y_pred_hour = y_pred[h]
        if len(np.unique(y_true_hour)) > 1:
            auc_val = roc_auc_score(y_true_hour, y_pred_hour)
            hourly_auc.append(auc_val)
            precision_vals, recall_vals, _ = precision_recall_curve(y_true_hour, y_pred_hour)
            prc_auc = auc(recall_vals, precision_vals)
            hourly_prc_auc.append(prc_auc)
        else:
            hourly_auc.append(np.nan)
            hourly_prc_auc.append(np.nan)
    return hourly_auc, hourly_prc_auc

# -----------------------------
# Custom Keras Callback for Metrics History
# -----------------------------
class MetricsHistory(Callback):
    """
    Custom Keras Callback to record evaluation metrics on the validation set after each epoch.
    Saves metrics (AUC, Accuracy, Precision, Recall, MCC) to a CSV file.
    """
    def __init__(self, X_val, y_val, path):
        super(MetricsHistory, self).__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.path = path
        self.results = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val)
        y_pred_bin = (y_pred > 0.5).astype(int)
        auc_val = roc_auc_score(self.y_val, y_pred)
        accuracy_val = accuracy_score(self.y_val, y_pred_bin)
        precision_val = precision_score(self.y_val, y_pred_bin, zero_division=0)
        recall_val = recall_score(self.y_val, y_pred_bin, zero_division=0)
        # Use sklearn's matthews_corrcoef for MCC
        from sklearn.metrics import matthews_corrcoef
        mcc_val = matthews_corrcoef(self.y_val, y_pred_bin)
        self.results.append({
            'epoch': epoch,
            'auc': auc_val,
            'accuracy': accuracy_val,
            'precision': precision_val,
            'recall': recall_val,
            'mcc': mcc_val
        })
        pd.DataFrame(self.results).to_csv(self.path, index=False)

# -----------------------------
# Calibration Functions
# -----------------------------
def calibrate_predictions(y_true, y_pred_prob):
    """
    Calibrate predictions using a calibration curve (probability binning).
    
    Returns:
        prob_true: True probabilities.
        prob_pred: Mean predicted probabilities.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10, strategy='uniform')
    return prob_true, prob_pred

def visualize_calibration(y_true, y_pred_prob, output_dir, fold):
    """
    Visualize calibration using CalibrationDisplay and save the plot.
    """
    disp = CalibrationDisplay.from_predictions(y_true, y_pred_prob, n_bins=10, strategy='uniform')
    plot_path = os.path.join(output_dir, f'calibration_plot_fold_{fold}.png')
    disp.figure_.savefig(plot_path)
    print(f"Calibration plot saved to {plot_path}")

