#!/usr/bin/env python3
"""
Test Performance Evaluation for Model vs. APACHE Comparison

This script:
  1. Loads APACHE and prediction CSV files.
  2. Merges them on ["patientid", "Time (Hours)"].
  3. Drops rows missing critical values ("APACHE", "Non-Survival Probability", "Outcome").
  4. Computes overall AUROC for the model and APACHE; inverts APACHE predictions if AUROC < 0.5.
  5. Plots the overall ROC curves.
  6. For each time point (e.g., 24, 72, 96, 120), selects for each patient the row closest to that time, computes bootstrap AUROC with CIs for both model and APACHE, and stores the results.
  7. Plots AUROC over time with error bars and annotations.
  
Usage:
    python test_performance_evaluation.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, matthews_corrcoef
from sklearn.utils import resample

# -----------------------------
# Global Time Points of Interest
# -----------------------------
TIME_POINTS = [24, 72, 96, 120]

# -----------------------------
# Function: Pick Closest Time
# -----------------------------
def pick_closest_time(group, t):
    """
    For a given patient group, pick the row with "Time (Hours)" closest to t.
    """
    idx = np.abs(group["Time (Hours)"] - t).idxmin()
    return group.loc[idx]

# -----------------------------
# Function: Bootstrap AUROC
# -----------------------------
def bootstrap_auc(y_true, y_pred, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    if len(bootstrapped_scores) == 0:
        return np.nan, np.nan, np.nan
    lower = np.percentile(bootstrapped_scores, 2.5)
    upper = np.percentile(bootstrapped_scores, 97.5)
    mean_score = np.mean(bootstrapped_scores)
    return lower, upper, mean_score

# -----------------------------
# Function: Fix APACHE Predictions If Needed
# -----------------------------
def fix_apache_if_needed(y_true, y_pred_apache):
    initial_auc = roc_auc_score(y_true, y_pred_apache)
    if initial_auc < 0.5:
        print(f"APACHE AUC = {initial_auc:.3f} < 0.5 -> Inverting predictions.")
        y_pred_apache = 1.0 - y_pred_apache
    return y_pred_apache

# -----------------------------
# Function: Load and Merge Data
# -----------------------------
def load_and_merge_data(apache_file, preds_file):
    """
    Loads APACHE data and model predictions, then merges them on ["patientid", "Time (Hours)"].
    """
    df_apache = pd.read_csv(apache_file)
    df_preds = pd.read_csv(preds_file)
    df_merged = pd.merge(df_apache, df_preds, how="inner", on=["patientid", "Time (Hours)"])
    print("Merged shape:", df_merged.shape)
    # Drop rows missing critical values
    required_cols = ["APACHE", "Non-Survival Probability", "Outcome"]
    df_merged.dropna(subset=required_cols, inplace=True)
    print("After dropping rows missing APACHE/model/outcome:", df_merged.shape)
    return df_merged

# -----------------------------
# Function: Evaluate Overall ROC and Plot
# -----------------------------
def evaluate_overall_roc(df_merged, output_folder):
    y_true = df_merged["Outcome"].values.astype(int)
    y_pred_model = df_merged["Non-Survival Probability"].values.astype(float)
    y_pred_apache = df_merged["APACHE"].values.astype(float)
    
    model_auc = roc_auc_score(y_true, y_pred_model)
    apache_auc = roc_auc_score(y_true, y_pred_apache)
    print(f"Model AUROC: {model_auc:.4f}")
    print(f"APACHE AUROC: {apache_auc:.4f}")
    
    if apache_auc < 0.5:
        print(f"APACHE AUROC ({apache_auc:.4f}) is less than 0.5. Inverting predictions.")
        y_pred_apache = 1.0 - y_pred_apache
        apache_auc = roc_auc_score(y_true, y_pred_apache)
        print(f"Inverted APACHE AUROC: {apache_auc:.4f}")
    
    # Compute ROC curves
    fpr_model, tpr_model, _ = roc_curve(y_true, y_pred_model)
    fpr_apache, tpr_apache, _ = roc_curve(y_true, y_pred_apache)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_model, tpr_model, label=f"Model (AUC = {model_auc:.2f})", color="blue")
    ax.plot(fpr_apache, tpr_apache, label=f"APACHE (AUC = {apache_auc:.2f})", color="red")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison: Model vs. APACHE")
    ax.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    roc_plot_file = os.path.join(output_folder, "ROC.png")
    fig.savefig(roc_plot_file)
    plt.show()
    print(f"ROC plot saved to: {roc_plot_file}")
    
    return y_true, y_pred_model, y_pred_apache

# -----------------------------
# Function: Evaluate Time Points with Bootstrap and Plot Error Bars
# -----------------------------
def evaluate_time_points(df_merged, time_points):
    model_auc_means = []
    model_errors = []  # (mean - lower, upper - mean)
    apache_auc_means = []
    apache_errors = []  # (mean - lower, upper - mean)
    
    for t in time_points:
        # For each patient, pick the row closest to time t
        df_t = df_merged.groupby("patientid", group_keys=False).apply(lambda group: pick_closest_time(group, t)).reset_index(drop=True)
        df_t.dropna(subset=["Outcome", "Non-Survival Probability", "APACHE"], inplace=True)
        
        if len(df_t) == 0:
            print(f"No valid data at/near hour {t}. Skipping.")
            model_auc_means.append(np.nan)
            model_errors.append((np.nan, np.nan))
            apache_auc_means.append(np.nan)
            apache_errors.append((np.nan, np.nan))
            continue
        
        y_true = df_t["Outcome"].astype(int).values
        y_pred_model = df_t["Non-Survival Probability"].astype(float).values
        y_pred_apache = df_t["APACHE"].astype(float).values
        y_pred_apache = fix_apache_if_needed(y_true, y_pred_apache)
        
        lb_model, ub_model, mean_model = bootstrap_auc(y_true, y_pred_model)
        lb_apache, ub_apache, mean_apache = bootstrap_auc(y_true, y_pred_apache)
        
        model_auc_means.append(mean_model)
        model_errors.append((mean_model - lb_model, ub_model - mean_model))
        apache_auc_means.append(mean_apache)
        apache_errors.append((mean_apache - lb_apache, ub_apache - mean_apache))
        
        print(f"Time {t} hrs -> Model AUROC: {mean_model:.3f} [{lb_model:.3f}, {ub_model:.3f}], "
              f"APACHE AUROC: {mean_apache:.3f} [{lb_apache:.3f}, {ub_apache:.3f}]")
    
    return model_auc_means, model_errors, apache_auc_means, apache_errors

def plot_timepoint_auc(time_points, model_auc_means, model_errors, apache_auc_means, apache_errors, output_folder):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(time_points, model_auc_means, yerr=np.transpose(model_errors), fmt='o-', color='blue', label='Model')
    ax.errorbar(time_points, apache_auc_means, yerr=np.transpose(apache_errors), fmt='s-', color='red', label='APACHE')
    ax.set_xlabel("Time (Hours)")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC Over Time (Model vs. APACHE)")
    ax.set_xticks(time_points)
    ax.set_ylim([0, 1])
    ax.legend(loc="lower right")
    
    for i, t in enumerate(time_points):
        mean_model = model_auc_means[i]
        err_low_model, err_high_model = model_errors[i]
        if not np.isnan(mean_model):
            text_model = f"{mean_model:.2f}\n[{mean_model - err_low_model:.2f}, {mean_model + err_high_model:.2f}]"
            ax.annotate(text_model, (t, mean_model), textcoords="offset points", xytext=(10,0),
                        ha='left', color="black", fontsize=9)
        mean_apache = apache_auc_means[i]
        err_low_apache, err_high_apache = apache_errors[i]
        if not np.isnan(mean_apache):
            text_apache = f"{mean_apache:.2f}\n[{mean_apache - err_low_apache:.2f}, {mean_apache + err_high_apache:.2f}]"
            ax.annotate(text_apache, (t, mean_apache), textcoords="offset points", xytext=(10,0),
                        ha='left', color="black", fontsize=9)
    
    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    roc_plot_file = os.path.join(output_folder, "auroc_time_points_errorbars_with_annotations.png")
    fig.savefig(roc_plot_file)
    plt.show()
    print(f"Timepoint AUROC plot saved to: {roc_plot_file}")

# -----------------------------
# Main Function
# -----------------------------
def main():
    # Define file paths for APACHE and predictions CSV files (update as needed)
    apache_file = "df_apache.csv"
    preds_file = "df_preds.csv"
    
    # Load and merge the data
    df_merged = load_and_merge_data(apache_file, preds_file)
    
    # Evaluate overall ROC and plot ROC curve
    overall_output_folder = "model_vs_apache"
    y_true, y_pred_model, y_pred_apache = evaluate_overall_roc(df_merged, overall_output_folder)
    plot_overall_roc(y_true, y_pred_model, y_pred_apache, overall_output_folder)
    
    # Evaluate AUROC at specific time points using bootstrapping
    model_auc_means, model_errors, apache_auc_means, apache_errors = evaluate_time_points(df_merged, TIME_POINTS)
    plot_timepoint_auc(TIME_POINTS, model_auc_means, model_errors, apache_auc_means, apache_errors, overall_output_folder)
    
if __name__ == "__main__":
    main()

