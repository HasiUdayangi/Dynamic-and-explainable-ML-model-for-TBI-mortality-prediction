
"""
Explainable Performance Evaluation for TBI Mortality Prediction
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_data():

    df_features = pd.read_csv("df_features.csv")
    df_importance = pd.read_csv("df_importance.csv")
    
    # Filter for non-negative Time_Step
    df_features = df_features[df_features["Time_Step"] >= 0].copy()
    df_preds = df_preds[df_preds["Time_Step"] >= 0].copy()
    
    # Merge features and predictions on "Time_Step"
    df_merged = pd.merge(df_features, df_preds, on="Time_Step", how="inner")
    print("Merged DataFrame columns:", df_merged.columns.tolist())
    return df_features, df_importance, df_preds, df_merged


def compute_effect_columns(df, features):
    """
    For each feature in the list, compute an effect column as the product of the feature value and its corresponding FI.
    """
    for feat in features:
        effect_col = "Effect_" + feat
        fi_col = feat + "_FI"
        if feat in df.columns and fi_col in df.columns:
            df[effect_col] = df[feat] * df[fi_col]
        else:
            df[effect_col] = 0.0
    return df

def compute_effects_in_merged(df, features):
    """Wrapper to compute effect columns on merged data."""
    return compute_effect_columns(df, features)

def top_positive_features_each_time(df, feature_list, n=5):
    df_sorted = df.sort_values("Time_Step")
    time_points = [t for t in sorted(df_sorted["Time_Step"].unique()) if t % 12 == 0]
    summary_dict = {}
    for t in time_points:
        row = df_sorted[df_sorted["Time_Step"] == t].iloc[0]
        pos_list = []
        for f in feature_list:
            eff = row.get(f"Effect_{f}", 0)
            if eff > 0:
                pos_list.append((f, eff))
        pos_list.sort(key=lambda x: x[1], reverse=True)
        summary_dict[t] = pos_list[:n]
    return summary_dict

def top_negative_features_each_time(df, feature_list, n=5):
    df_sorted = df.sort_values("Time_Step")
    time_points = [t for t in sorted(df_sorted["Time_Step"].unique()) if t % 12 == 0]
    summary_dict = {}
    for t in time_points:
        row = df_sorted[df_sorted["Time_Step"] == t].iloc[0]
        neg_list = []
        for f in feature_list:
            eff = row.get(f"Effect_{f}", 0)
            if eff < 0:
                neg_list.append((f, eff))
        neg_list.sort(key=lambda x: x[1])  # most negative first
        summary_dict[t] = neg_list[:n]
    return summary_dict

def top_positive_features_each_time_merged(df, feature_list, n=5):
    # For merged data with "Time" column; adjust if needed.
    df_sorted = df.sort_values("Time")
    time_points = [t for t in sorted(df_sorted["Time"].unique()) if t % 12 == 0]
    summary_dict = {}
    for t in time_points:
        row = df_sorted[df_sorted["Time"] == t].iloc[0]
        pos_list = []
        for f in feature_list:
            eff = row.get(f"Effect_{f}", 0)
            if eff > 0:
                pos_list.append((f, eff))
        pos_list.sort(key=lambda x: x[1], reverse=True)
        summary_dict[t] = pos_list[:n]
    return summary_dict

def top_negative_features_each_time_merged(df, feature_list, n=5):
    df_sorted = df.sort_values("Time")
    time_points = [t for t in sorted(df_sorted["Time"].unique()) if t % 12 == 0]
    summary_dict = {}
    for t in time_points:
        row = df_sorted[df_sorted["Time"] == t].iloc[0]
        neg_list = []
        for f in feature_list:
            eff = row.get(f"Effect_{f}", 0)
            if eff < 0:
                neg_list.append((f, eff))
        neg_list.sort(key=lambda x: x[1])
        summary_dict[t] = neg_list[:n]
    return summary_dict


def get_text_color(hex_color):
    # For simplicity, always return black text.
    return "black"

def map_color_non_survival(eff, local_min, local_max):
    if abs(local_max - local_min) < 1e-12:
        return "#ffffff"
    ratio = (eff - local_min) / (local_max - local_min)
    ratio = min(max(ratio, 0.0), 1.0)
    light_red = np.array([255, 220, 220])
    dark_red  = np.array([180, 40, 40])
    color_rgb = light_red + (dark_red - light_red) * ratio
    return "#%02x%02x%02x" % (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))

def map_color_survival(eff, top_list):
    neg_abs_vals = [abs(val) for (_, val) in top_list]
    if not neg_abs_vals:
        return "#ffffff"
    max_abs = max(neg_abs_vals)
    if max_abs < 1e-12:
        return "#ffffff"
    ratio = abs(eff) / max_abs
    ratio = min(max(ratio, 0.0), 1.0)
    light_blue = np.array([190, 220, 230])
    dark_blue  = np.array([30, 70, 150])
    color_rgb = light_blue + (dark_blue - light_blue) * ratio
    return "#%02x%02x%02x" % (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))

def plot_script_chart(summary_dict, n=5, title="", use_survival_colormap=False, save_path=None):
    time_points = sorted(summary_dict.keys())
    num_cols = len(time_points)
    num_rows = n
    
    fig, ax = plt.subplots(figsize=(num_cols * 1.2, num_rows * 0.8))
    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    
    for col_idx, t in enumerate(time_points):
        top_list = summary_dict[t]
        if not use_survival_colormap:
            if top_list:
                local_min = min(val for (_, val) in top_list)
                local_max = max(val for (_, val) in top_list)
            else:
                local_min, local_max = 0, 1
        for row_idx in range(num_rows):
            x = col_idx
            y = num_rows - row_idx - 1
            cell_text = ""
            cell_color = "#ffffff"
            if row_idx < len(top_list):
                feat, eff = top_list[row_idx]
                cell_text = feat
                if use_survival_colormap:
                    cell_color = map_color_survival(eff, top_list)
                else:
                    cell_color = map_color_non_survival(eff, local_min, local_max)
            rect = Rectangle((x, y), 1, 1, facecolor=cell_color, edgecolor="white")
            ax.add_patch(rect)
            txt_color = get_text_color(cell_color)
            ax.text(x + 0.5, y + 0.5, cell_text, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=txt_color)
    
    ax.set_xticks([i + 0.5 for i in range(num_cols)])
    ax.set_xticklabels(time_points, fontsize=12, fontweight="bold")
    ax.set_yticks([])
    
    for row_idx in range(num_rows):
        rank_label = row_idx + 1
        y_pos = num_rows - row_idx - 0.5
        ax.text(num_cols + 0.1, y_pos, f"Rank {rank_label}",
                ha="left", va="center", fontsize=7, fontweight="bold")
    
    ax.set_xlabel("Time (Hours)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    plt.show()
    return fig

def load_apache_data(apache_filepath, patient_id):
    df_apache = pd.read_csv(apache_filepath)
    df_patient = df_apache[df_apache["patientid"] == patient_id].copy()
    if "Time_Step" not in df_patient.columns and "hour" in df_patient.columns:
        df_patient.rename(columns={"hour": "Time_Step"}, inplace=True)
    df_patient = df_patient[df_patient["Time_Step"] >= 2].copy()
    return df_patient


def filter_merged_data(df, lower=2, upper=72):
    return df[(df["Time_Step"] >= lower) & (df["Time_Step"] <= upper)].copy()


def compute_sum_effects(df, effect_cols):
    df["sum_pos_effect"] = df[effect_cols].apply(lambda row: sum(val for val in row if val > 0), axis=1)
    df["sum_neg_effect"] = df[effect_cols].apply(lambda row: sum(val for val in row if val < 0), axis=1)
    return df

def plot_fill_between(df_plot, df_apache_patient, output_folder, patient_id):
    fig, ax = plt.subplots(figsize=(12, 6))
    y_pred = df_plot["Non-Survival Probability"]
    ax.plot(df_plot["Time_Step"], y_pred, color="black", linewidth=2, label="Model predictions")
    
    if df_apache_patient is not None and not df_apache_patient.empty and "APACHE" in df_apache_patient.columns:
        ax.plot(df_apache_patient["Time_Step"], df_apache_patient["APACHE"],
                "r--", linewidth=2, label="APACHE Score")
    else:
        print("APACHE data not found or empty for patient.")
    
    ax.fill_between(
        df_plot["Time_Step"],
        y_pred,
        y_pred + df_plot["sum_pos_effect"],
        where=(df_plot["sum_pos_effect"] > 0),
        color="red", alpha=0.3,
        label="Features driving non-survival"
    )
    ax.fill_between(
        df_plot["Time_Step"],
        y_pred + df_plot["sum_neg_effect"],
        y_pred,
        where=(df_plot["sum_neg_effect"] < 0),
        color="royalblue", alpha=0.3,
        label="Features driving survival"
    )
    
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time Since Admission (Hours)")
    ax.set_ylabel("Probability of Non-Survival")
    ax.set_title(f"Fill-Between Chart for Patient {patient_id}")
    xticks = np.arange(0, 160, 12)
    ax.set_xticks(xticks)
    
    for t in xticks:
        row = df_plot.iloc[(df_plot["Time_Step"] - t).abs().argsort()[:1]]
        if not row.empty:
            pred_val = row["Non-Survival Probability"].values[0]
            ax.annotate(f"P: {pred_val:.2f}", xy=(t, pred_val),
                        xytext=(0,5), textcoords="offset points", color="black", fontsize=9)
        if df_apache_patient is not None and not df_apache_patient.empty and "APACHE" in df_apache_patient.columns:
            row_a = df_apache_patient.iloc[(df_apache_patient["Time_Step"] - t).abs().argsort()[:1]]
            if not row_a.empty:
                apache_val = row_a["APACHE"].values[0]
                ax.annotate(f"A: {apache_val:.2f}", xy=(t, apache_val),
                            xytext=(0,-15), textcoords="offset points", color="red", fontsize=9)
    
    ax.legend(loc="upper right")
    plt.tight_layout()
    plot_filename = os.path.join(output_folder, f"{patient_id}_survival_non_effect.png")
    plt.savefig(plot_filename)
    plt.show()
    print("Fill-Between plot saved to:", plot_filename)
    return fig

def merge_features_importance(df_features, df_importance):
    df_merged2 = pd.merge(df_features, df_importance, on="Time", how="inner")
    print("Merged DataFrame columns from df_features and df_importance:")
    print(df_merged2.columns.tolist())
    return df_merged2

def compute_effects_in_merged(df, features):
    return compute_effect_columns(df, features)

def top_positive_features_each_time_merged(df, feature_list, n=5):
    df_sorted = df.sort_values("Time")
    time_points = [t for t in sorted(df_sorted["Time"].unique()) if t % 12 == 0]
    summary_dict = {}
    for t in time_points:
        row = df_sorted[df_sorted["Time"] == t].iloc[0]
        pos_list = []
        for f in feature_list:
            eff = row.get(f"Effect_{f}", 0)
            if eff > 0:
                pos_list.append((f, eff))
        pos_list.sort(key=lambda x: x[1], reverse=True)
        summary_dict[t] = pos_list[:n]
    return summary_dict

def top_negative_features_each_time_merged(df, feature_list, n=5):
    df_sorted = df.sort_values("Time")
    time_points = [t for t in sorted(df_sorted["Time"].unique()) if t % 12 == 0]
    summary_dict = {}
    for t in time_points:
        row = df_sorted[df_sorted["Time"] == t].iloc[0]
        neg_list = []
        for f in feature_list:
            eff = row.get(f"Effect_{f}", 0)
            if eff < 0:
                neg_list.append((f, eff))
        neg_list.sort(key=lambda x: x[1])
        summary_dict[t] = neg_list[:n]
    return summary_dict

def process_patient(pid, test_data_df, apache_df, best_model, embedding_dim=16, max_seq_len=72):
    """
    Processes data for one patient:
      - Filters the test data and APACHE data for the patient.
      - Preprocesses test data to generate hourly feature vectors.
      - Predicts hourly non-survival probability using the best model.
      - Merges the predictions with the patient's APACHE values.
      
    Returns:
      A merged DataFrame with columns "patientid", "Time (Hours)", "Non-Survival Probability", and APACHE.
    """
    # Filter test data for the given patient ID
    patient_data = test_data_df[test_data_df["patientid"] == pid].copy()
    if patient_data.empty:
        print(f"No test data found for patient {pid}.")
        return None
    
    # Preprocess data to get hourly sequences
    hourly_sequences = preprocess_patient_data_hourly(patient_data, embedding_dim, max_seq_len)
    print(f"Patient {pid}: Hourly sequence shape: {hourly_sequences.shape}")
    
    # Predict hourly probabilities using the best model
    hourly_probs = []
    for seq in hourly_sequences:
        # Expand dims to create a batch of 1
        prob = best_model.predict(np.expand_dims(seq, axis=0)).flatten()[0]
        hourly_probs.append(prob)
    
    # Create a DataFrame with model predictions
    df_preds = pd.DataFrame({
        "Time (Hours)": np.arange(max_seq_len),
        "Non-Survival Probability": hourly_probs,
        "patientid": pid
    })
    
    # Filter APACHE data for the patient
    df_apache_patient = apache_df[apache_df["patientid"] == pid].copy()
    if "Time (Hours)" not in df_apache_patient.columns and "hour" in df_apache_patient.columns:
        df_apache_patient.rename(columns={"hour": "Time (Hours)"}, inplace=True)
    
    if df_apache_patient.empty:
        print(f"No APACHE data found for patient {pid}. Returning model predictions only.")
        return df_preds
    
    # Merge model predictions with APACHE values on ["patientid", "Time (Hours)"]
    df_merged = pd.merge(df_preds, df_apache_patient, on=["patientid", "Time (Hours)"], how="inner")
    return df_merged

def process_patient(pid, test_data_df, apache_df, best_model, embedding_dim=16, max_seq_len=72):

    # Filter test data for the given patient ID
    patient_data = test_data_df[test_data_df["patientid"] == pid].copy()
    if patient_data.empty:
        print(f"No test data found for patient {pid}.")
        return None
    
    # Preprocess data to get hourly sequences
    hourly_sequences = preprocess_patient_data_hourly(patient_data, embedding_dim, max_seq_len)
    print(f"Patient {pid}: Hourly sequence shape: {hourly_sequences.shape}")
    
    # Predict hourly probabilities using the best model
    hourly_probs = []
    for seq in hourly_sequences:
        # Expand dims to create a batch of 1
        prob = best_model.predict(np.expand_dims(seq, axis=0)).flatten()[0]
        hourly_probs.append(prob)
    
    # Create a DataFrame with model predictions
    df_preds = pd.DataFrame({
        "Time (Hours)": np.arange(max_seq_len),
        "Non-Survival Probability": hourly_probs,
        "patientid": pid
    })
    
    # Filter APACHE data for the patient
    df_apache_patient = apache_df[apache_df["patientid"] == pid].copy()
    if "Time (Hours)" not in df_apache_patient.columns and "hour" in df_apache_patient.columns:
        df_apache_patient.rename(columns={"hour": "Time (Hours)"}, inplace=True)
    
    if df_apache_patient.empty:
        print(f"No APACHE data found for patient {pid}. Returning model predictions only.")
        return df_preds
    
    # Merge model predictions with APACHE values on ["patientid", "Time (Hours)"]
    df_merged = pd.merge(df_preds, df_apache_patient, on=["patientid", "Time (Hours)"], how="inner")
    return df_merged


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


def evaluate_time_points(df_merged, time_points):
    model_auc_means = []
    model_errors = []  # (mean - lower, upper - mean)
    apache_auc_means = []
    apache_errors = []  # (mean - lower, upper - mean)
    
    for t in time_points:
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


# ==========================
# Main Function: Process Patient Data, Run Model, Merge with APACHE, and Evaluate
# ==========================
def main():
    test_data_file = ""           # CSV with test data for multiple patients
    apache_file = ""              # CSV with APACHE values for multiple patients
    best_model_path = ""
    output_predictions_file = ""
    
    # For additional explainability analysis:
    features = [
        "HR_min", "HR_max", "HR_median",
        "DBP_min", "DBP_max", "DBP_median",
        "SBP_min", "SBP_max", "SBP_median",
        "ABP_mean_min", "ABP_mean_max", "ABP_mean_median",
        "SpO2_min", "SpO2_max", "SpO2_median",
        "Age"
    ]
    
    # Patient-specific parameters for explainability plots:
    patient_id = ""  # Update with desired patient ID
    output_folder = ""
    os.makedirs(output_folder, exist_ok=True)
    apache_filepath = ""
    
    # ----- Load Data -----
    test_data_df = pd.read_csv(test_data_file)
    apache_df = pd.read_csv(apache_file)
    print(f"Loaded test data with shape: {test_data_df.shape}")
    print(f"Loaded APACHE data with shape: {apache_df.shape}")
    
    # Load best model
    best_model = load_model(best_model_path)
    
    # Process each patient in the test set (ensure predictions are kept separate per patient)
    patient_ids = test_data_df["patientid"].unique()
    print(f"Found {len(patient_ids)} patients in test data.")
    all_merged = []
    
    for pid in patient_ids:
        print(f"Processing patient {pid}...")
        df_merged_patient = process_patient(pid, test_data_df, apache_df, best_model, embedding_dim=16, max_seq_len=72)
        if df_merged_patient is not None:
            all_merged.append(df_merged_patient)
    
    if all_merged:
        combined_df = pd.concat(all_merged, ignore_index=True)
        combined_df.to_csv(output_predictions_file, index=False)
        print(f"Combined predictions saved to: {output_predictions_file}")
    else:
        print("No patient data processed.")
        return
    
    # ----- Overall Evaluation for Combined Data -----
    overall_output_folder = ""
    os.makedirs(overall_output_folder, exist_ok=True)
    y_true, y_pred_model, y_pred_apache = evaluate_overall_roc(combined_df, overall_output_folder)
    
    # Evaluate AUROC at specific time points using bootstrapping and plot error bars
    model_auc_means, model_errors, apache_auc_means, apache_errors = evaluate_time_points(combined_df, TIME_POINTS)
    plot_timepoint_auc(TIME_POINTS, model_auc_means, model_errors, apache_auc_means, apache_errors, overall_output_folder)
    
    # ----- Additional Explainability Analysis -----
    # Merge df_features and df_importance for additional feature effect analysis
    df_features = pd.read_csv("")
    df_importance = pd.read_csv("")
    df_merged2 = merge_features_importance(df_features, df_importance)
    df_merged2 = compute_effects_in_merged(df_merged2, features)
    
    # Filter merged predictions (df_merged) for the patient and desired time range
    df_patient = combined_df[combined_df["patientid"] == patient_id].copy()
    df_plot = df_patient[(df_patient["Time_Step"] >= 2) & (df_patient["Time_Step"] <= 72)].copy()
    
    # Compute sum of positive and negative effects in df_plot
    all_effect_cols = [c for c in df_plot.columns if c.startswith("Effect_")]
    df_plot = compute_sum_effects(df_plot, all_effect_cols)
    
    # Load APACHE data for the patient for explainability analysis
    df_apache_patient = load_apache_data(apache_filepath, patient_id)
    
    # Plot fill-between chart with annotations for the patient
    fig_fill = plot_fill_between(df_plot, df_apache_patient, output_folder, patient_id)
    
    # Compute summary dictionaries from merged features & importance (df_merged2)
    summary_dict_pos = top_positive_features_each_time_merged(df_merged2, features, n=5)
    summary_dict_neg = top_negative_features_each_time_merged(df_merged2, features, n=5)
    
    # Define output folder for script charts (if different, update accordingly)
    output_folder_effects = ""
    os.makedirs(output_folder_effects, exist_ok=True)
    
    save_path_pos = os.path.join(output_folder_effects, "")
    save_path_neg = os.path.join(output_folder_effects, "")
    
    fig_pos = plot_script_chart(summary_dict_pos, n=5, 
                                title="Top Non-Survival Feature Effects",
                                use_survival_colormap=False, 
                                save_path=save_path_pos)
    
    fig_neg = plot_script_chart(summary_dict_neg, n=5, 
                                title="Top Survival Feature Effects",
                                use_survival_colormap=True, 
                                save_path=save_path_neg)
    
    # Save the merged features-importance dataframe for further analysis
    merged_save_path = ""
    df_merged2.to_csv(merged_save_path, index=False)
    print("Merged dataframe saved as", merged_save_path)

if __name__ == "__main__":
    main()
