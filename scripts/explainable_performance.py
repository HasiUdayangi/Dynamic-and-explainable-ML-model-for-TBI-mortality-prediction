
"""
Explainable Performance Evaluation for TBI Mortality Prediction
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ==========================
# Part 1: Data Loading and Merging
# ==========================
def load_data():
    """
    Load df_features, df_importance, and df_preds from CSV files,
    filter for non-negative Time_Step, and merge features with predictions.
    """
    df_features = pd.read_csv("df_features.csv")
    df_importance = pd.read_csv("df_importance.csv")
    df_preds = pd.read_csv("df_preds.csv")
    
    # Filter rows with non-negative Time_Step
    df_features = df_features[df_features["Time_Step"] >= 0].copy()
    df_preds = df_preds[df_preds["Time_Step"] >= 0].copy()
    
    # Merge features and predictions on "Time_Step"
    df_merged = pd.merge(df_features, df_preds, on="Time_Step", how="inner")
    print("Merged DataFrame columns:", df_merged.columns.tolist())
    return df_features, df_importance, df_preds, df_merged
   

# ==========================
# Part 2: Compute Effect Columns
# ==========================
def compute_effect_columns(df, features):
    """
    For each feature in the list, compute an effect column as the product
    of the feature value and its corresponding FI value.
    """
    for feat in features:
        effect_col = "Effect_" + feat
        fi_col = feat + "_FI"
        if feat in df.columns and fi_col in df.columns:
            df[effect_col] = df[feat] * df[fi_col]
        else:
            df[effect_col] = 0.0
    return df

# ==========================
# Part 3: Summary Dictionary Functions
# ==========================
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

# ==========================
# Part 4: Color Mapping and Plotting Function
# ==========================
def get_text_color(hex_color):
    # For simplicity, always return black
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

# ==========================
# Part 5: Load APACHE Data
# ==========================
def load_apache_data(apache_filepath, patient_id):
    df_apache = pd.read_csv(apache_filepath)
    df_patient = df_apache[df_apache["patientid"] == patient_id].copy()
    if "Time_Step" not in df_patient.columns and "hour" in df_patient.columns:
        df_patient.rename(columns={"hour": "Time_Step"}, inplace=True)
    df_patient = df_patient[df_patient["Time_Step"] >= 2].copy()
    return df_patient

# ==========================
# Part 6: Filter Merged Data for Desired Time Range
# ==========================
def filter_merged_data(df, lower=2, upper=72):
    return df[(df["Time_Step"] >= lower) & (df["Time_Step"] <= upper)].copy()

# ==========================
# Part 7: Compute Sum of Positive and Negative Effects
# ==========================
def compute_sum_effects(df, effect_cols):
    df["sum_pos_effect"] = df[effect_cols].apply(lambda row: sum(val for val in row if val > 0), axis=1)
    df["sum_neg_effect"] = df[effect_cols].apply(lambda row: sum(val for val in row if val < 0), axis=1)
    return df

# ==========================
# Part 8: Plot Fill-Between Chart with Annotations
# ==========================
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

# ==========================
# Part 9: Merge df_features and df_importance for Additional Analysis
# ==========================
def merge_features_importance(df_features, df_importance):
    df_merged2 = pd.merge(df_features, df_importance, on="Time", how="inner")
    print("Merged DataFrame columns from df_features and df_importance:")
    print(df_merged2.columns.tolist())
    return df_merged2

def compute_effects_in_merged(df, features):
    for feat in features:
        effect_col = "Effect_" + feat
        fi_col = feat + "_FI"
        if feat in df.columns and fi_col in df.columns:
            df[effect_col] = df[feat] * df[fi_col]
        else:
            df[effect_col] = 0.0
    return df

# ==========================
# Part 10: Summary Dictionaries for Top Features from Merged Data
# ==========================
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

# ==========================
# Main Function
# ==========================
def main():
    # Load merged dataframe (df_merged) from previous steps
    # For demonstration, we assume df_merged is available as "SHAP_final_analysis_merged.csv"
    df_merged = pd.read_csv("SHAP_final_analysis_merged.csv")
    
    # Load additional data for merging
    df_features = pd.read_csv("df_features.csv")
    df_importance = pd.read_csv("df_importance.csv")
    
    # Define list of features
    features = [
        "HR_min", "HR_max", "HR_median",
        "DBP_min", "DBP_max", "DBP_median",
        "SBP_min", "SBP_max", "SBP_median",
        "ABP_mean_min", "ABP_mean_max", "ABP_mean_median",
        "SpO2_min", "SpO2_max", "SpO2_median",
        "Age"
    ]
    
    # ---------------------------
    # Step 1: Merge DataFrames for Additional Analysis
    df_merged2 = merge_features_importance(df_features, df_importance)
    df_merged2 = compute_effects_in_merged(df_merged2, features)
    
    # ---------------------------
    # Step 2: Compute and Plot Feature Effects for a Patient
    # Filter df_merged for desired time range (2–72 Hours)
    df_plot = df_merged[(df_merged["Time_Step"] >= 2) & (df_merged["Time_Step"] <= 72)].copy()
    # Compute sum of positive and negative effects
    all_effect_cols = [c for c in df_plot.columns if c.startswith("Effect_")]
    df_plot["sum_pos_effect"] = df_plot[all_effect_cols].apply(lambda row: sum(val for val in row if val > 0), axis=1)
    df_plot["sum_neg_effect"] = df_plot[all_effect_cols].apply(lambda row: sum(val for val in row if val < 0), axis=1)
    
    patient_id = ""
    output_folder = ""
    os.makedirs(output_folder, exist_ok=True)
    # Load APACHE data for the patient
    apache_file = ""
    df_apache_patient = pd.read_csv(apache_file)
    df_apache_patient = df_apache_patient[df_apache_patient["patientid"] == patient_id].copy()
    if "Time_Step" not in df_apache_patient.columns and "hour" in df_apache_patient.columns:
        df_apache_patient.rename(columns={"hour": "Time_Step"}, inplace=True)
    df_apache_patient = df_apache_patient[df_apache_patient["Time_Step"] >= 2].copy()
    
    # Plot fill-between chart
    fig_fill = plot_fill_between(df_plot, df_apache_patient, output_folder, patient_id)
    
    # ---------------------------
    # Step 3: Compute Summary Dictionaries and Plot Script Charts for Merged Data
    summary_dict_pos = top_positive_features_each_time_merged(df_merged2, features, n=5)
    summary_dict_neg = top_negative_features_each_time_merged(df_merged2, features, n=5)
    
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
    
    # ---------------------------
    # Step 4: Save the Merged DataFrame for Further Analysis
    df_merged2.to_csv("", index=False)
    print("Merged dataframe saved as SHAP_final_analysis_merged.csv")
    
if __name__ == "__main__":
    main()

