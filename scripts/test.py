

import os
import numpy as np
import pandas as pd
from IPython.display import clear_output
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.calibration import calibration_curve, CalibrationDisplay
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Define output directory and ensure it exists
output_dir = ""
os.makedirs(output_dir, exist_ok=True)

# Define file paths for saving results
results_filepath = os.path.join(output_dir, '')
calibration_test_path = os.path.join(output_dir, '')
calibration_plot_path = os.path.join(output_dir, '')

# Load test data and trained model (update paths if needed)
X_test = np.load('/X_test.npy')
y_test = np.load('/y_test.npy')
model = load_model('')

# Evaluate the model for each hour slice
hourly_test_metrics = []
for hour in range(1, X_test.shape[1] + 1):
    # Slice the test data up to the current hour
    X_test_hour = X_test[:, :hour, :]
    
    # Get model predictions for the sliced data
    y_pred_test_hour = model.predict(X_test_hour).flatten()
    
    try:
        auc_test = roc_auc_score(y_test, y_pred_test_hour)
    except Exception as e:
        auc_test = np.nan
    accuracy_test = accuracy_score(y_test, y_pred_test_hour > 0.5)
    precision_test = precision_score(y_test, y_pred_test_hour > 0.5, zero_division=0)
    recall_test = recall_score(y_test, y_pred_test_hour > 0.5, zero_division=0)
    mcc_test = matthews_corrcoef(y_test, y_pred_test_hour > 0.5)
    
    hourly_test_metrics.append({
        'hour': hour,
        'auc': auc_test,
        'accuracy': accuracy_test,
        'precision': precision_test,
        'recall': recall_test,
        'mcc': mcc_test
    })

clear_output(wait=True)
# Save hourly test metrics to a DataFrame and CSV file
hourly_test_metrics_df = pd.DataFrame(hourly_test_metrics)
hourly_test_metrics_df.to_csv(results_filepath, index=False)
print(f"Hourly test metrics saved to {results_filepath}")

# Full test predictions and calibration
y_pred_test_full = model.predict(X_test).flatten()  # Full test predictions
prob_true, prob_pred = calibration_curve(y_test, y_pred_test_full, n_bins=10, strategy='uniform')

# Save calibration results to CSV
calibration_test_df = pd.DataFrame({
    'prob_true': prob_true,
    'prob_pred': prob_pred
})
calibration_test_df.to_csv(calibration_test_path, index=False)
print(f"Calibration results saved to {calibration_test_path}")

# Generate and save calibration plot
disp = CalibrationDisplay.from_predictions(y_test, y_pred_test_full, n_bins=10, strategy='uniform')
disp.figure_.savefig(calibration_plot_path)
print(f"Calibration plot saved to {calibration_plot_path}")

