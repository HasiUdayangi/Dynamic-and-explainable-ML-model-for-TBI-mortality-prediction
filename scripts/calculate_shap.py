#!/usr/bin/env python3
"""
Calculate SHAP Values and Generate Summary Plots for TBI HRV Prediction Model

This script:
  - Loads training data (X_train_resampled) and test data (X_test) from .npy files.
  - Loads the best trained model.
  - Constructs a fixed-input model by applying a Lambda layer to squeeze the output dimension.
  - Uses SHAP GradientExplainer to compute SHAP values for the test data.
  - Saves the computed SHAP values as both .npy and .pkl files.
  - Generates summary plots for selected time steps (24, 48, and 120) and for the time-averaged values.
  - Saves all plots to an output folder.
  
Usage:
    python calculate_shap.py
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Lambda
from IPython.display import clear_output
import shap

# ------------------------------
# Configuration – Update paths as needed!
# ------------------------------
# Path to saved training and test data
X_train_path = 'data/token/X_train_resampled.npy'
y_train_path = 'data/token/y_train_resampled.npy'
X_test_path = 'data/token/X_test.npy'

# Path to best model file (ensure the file path is correct)
best_model_path = "real time output hourly token_final_moreSHAP/best_model_BiLSTM_cumulative_2.keras"

# Output folder for SHAP analysis results
output_folder = "SHAP final analysis"
os.makedirs(output_folder, exist_ok=True)

# ------------------------------
# Load Data and Model
# ------------------------------
X_train_resampled = np.load(X_train_path)
X_test = np.load(X_test_path)
best_model = load_model(best_model_path)

# ------------------------------
# Create a fixed-input model to remove extra dimensions
# (Assuming the model expects input shape (689,16); update if needed)
fixed_input = Input(shape=(689, 16))
fixed_output = Lambda(lambda x: tf.squeeze(x, axis=-1))(best_model(fixed_input))
fixed_model = Model(inputs=fixed_input, outputs=fixed_output)
print("New model input shape:", fixed_model.input_shape)  # Should be (None, 689, 16)

# ------------------------------
# Initialize SHAP GradientExplainer and Compute SHAP Values
# ------------------------------
print("Initializing SHAP GradientExplainer...")
explainer = shap.GradientExplainer(fixed_model, X_train_resampled)
print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_test)
print("SHAP values shape (before squeezing):", np.array(shap_values).shape)

# Squeeze extra dimension (assuming output shape becomes (num_samples, 689, 16))
shap_values = np.squeeze(shap_values, axis=-1)
print("SHAP values shape (after squeezing):", shap_values.shape)

# Save SHAP values using NumPy and Pickle
np.save(os.path.join(output_folder, 'shap_values.npy'), shap_values)
with open(os.path.join(output_folder, 'shap_values.pkl'), 'wb') as f:
    pickle.dump(shap_values, f)
print("SHAP values saved successfully.")

# ------------------------------
# Define Feature Names for Summary Plots
# ------------------------------
features = [
    "HR_min", "HR_max", "HR_median", "DBP_min", "DBP_max", "DBP_median",
    "SBP_min", "SBP_max", "SBP_median", "ABP_mean_min", "ABP_mean_max",
    "ABP_mean_median", "SpO2_min", "SpO2_max", "SpO2_median", "Age"
]

# ------------------------------
# Generate Summary Plots for Selected Time Steps
# ------------------------------
# List of time steps to analyze (e.g., 24, 48, 120)
time_steps = [24, 48, 120]

for time_step in time_steps:
    # Extract SHAP values and corresponding test data for the time step
    shap_values_time = shap_values[:, time_step, :]  # Expected shape: (num_samples, 16)
    X_test_time = X_test[:, time_step, :]             # Expected shape: (num_samples, 16)
    
    # Generate standard summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values_time, X_test_time, feature_names=features, show=False)
    plot_filename = os.path.join(output_folder, f"shap_summary_time_{time_step}.png")
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    print(f"Summary plot (beeswarm) saved to: {plot_filename}")
    
    # Generate bar plot summary
    plt.figure()
    shap.summary_plot(shap_values_time, X_test_time, feature_names=features, plot_type="bar", show=False)
    plot_filename = os.path.join(output_folder, f"summary_plot_time_{time_step}.png")
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    print(f"Bar summary plot saved to: {plot_filename}")

# ------------------------------
# Generate Summary Plots for Averaged SHAP Values
# ------------------------------
# Average over the time dimension for each sample
X_test_avg = np.mean(X_test, axis=1)         # Shape: (num_samples, 16)
shap_values_avg = np.mean(shap_values, axis=1) # Shape: (num_samples, 16)

# Generate beeswarm summary plot for averaged values
plt.figure()
shap.summary_plot(shap_values_avg, X_test_avg, feature_names=features, show=False)
plot_filename = os.path.join(output_folder, f"shap_summary_time_overtime.png")
plt.savefig(plot_filename, bbox_inches='tight')
plt.close()
print("Summary plot (beeswarm, averaged) saved to:", plot_filename)

# Generate bar plot summary for averaged values
plt.figure()
shap.summary_plot(shap_values_avg, X_test_avg, feature_names=features, plot_type="bar", show=False)
plot_filename = os.path.join(output_folder, f"summary_plot_time_overtime.png")
plt.savefig(plot_filename, bbox_inches='tight')
plt.close()
print("Bar summary plot (averaged) saved to:", plot_filename)

