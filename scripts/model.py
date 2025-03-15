#!/usr/bin/env python3
"""
Model Training and Evaluation for TBI HRV Prediction
"""

import os
import pickle
import itertools
import numpy as np
import pandas as pd

# TensorFlow/Keras imports
from tensorflow.keras.layers import Input, Masking, Bidirectional, LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow.keras.backend as K

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef

# Import evaluation functions from evaluation.py (ensure these functions exist in that file)
from evaluation import calibrate_predictions, calculate_hourly_metrics, bootstrap_metrics, matthews
from sklearn.calibration import CalibrationDisplay

# ------------------------------
# Load previously saved training data (update paths as needed)
# ------------------------------
X_train_resampled = np.load('/X_resampled_new.npy')
y_train_resampled = np.load('/y_resampled_new.npy')

# ------------------------------
# Define output paths and configuration
# ------------------------------
output_dir = "real time output hourly token_final"
batch_size = 1
os.makedirs(output_dir, exist_ok=True)

results_filepath = os.path.join(output_dir, 'results.csv')
cv_results_path = os.path.join(output_dir, 'CV_results_gridsearch.csv')
bootstrap_results_path = os.path.join(output_dir, 'bootstrap_results.csv')
best_model_path = os.path.join(output_dir, 'best_model_BiLSTM_cumulative_2.keras')
results_calibration = os.path.join(output_dir, 'hourly_predictions.csv')

bootstrap_df = pd.DataFrame()
hourly_metrics = []

# Parameter grid for grid search
param_dict = {
    'activation': ['sigmoid'],
    'dropout': [0.2, 0.3],
    'units': [16, 32, 64, 128],
    'layers': [2],
    'optimizer': ['adam']
}

AUCheader = ['auc_train', 'auc_val', 'matthews_train', 'matthews_val', 'cv_num'] + list(param_dict.keys())
CVheader = ['acc', 'loss', 'matthews', 'precision', 'recall', 'val_acc', 'val_loss', 'val_matthews', 'val_precision', 'val_recall', 'cv_num'] + list(param_dict.keys())

# ------------------------------
# Define model creation function
# ------------------------------
def create_model(features_per_timestep, units, activation, dropout, optimizer, layers):
    input_layer = Input(shape=(None, features_per_timestep))  # variable sequence length
    x = Masking(mask_value=0.0)(input_layer)
    if layers == 1:
        lstm = Bidirectional(LSTM(units, activation=activation, recurrent_dropout=dropout))(x)
    else:
        lstm1 = Bidirectional(LSTM(units, return_sequences=True, activation=activation, recurrent_dropout=dropout))(x)
        lstm = Bidirectional(LSTM(units, activation=activation, recurrent_dropout=dropout))(lstm1)
    output_layer = Dense(1, activation='sigmoid')(lstm)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', 'AUC'])
    return model

# ------------------------------
# Cross-validation setup
# ------------------------------
n_splits = 5 
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def append_results_to_csv(df, file_path):
    with open(file_path, 'a') as f:
        df.to_csv(f, sep='\t', index=False, header=f.tell() == 0)

auc_records = []
cv_records = []
best_auc = float('-inf')

calibrated_val_preds = []
calibrated_metrics = []

# ------------------------------
# Begin Grid Search and Cross-validation
# ------------------------------
for config in itertools.product(*param_dict.values()):
    config = dict(zip(param_dict.keys(), config))
    print("Testing config:", config)
    
    for i, (train_idx, val_idx) in enumerate(kfold.split(X_train_resampled, y_train_resampled)):
        X_train = X_train_resampled[train_idx]
        X_val = X_train_resampled[val_idx]
        y_train = y_train_resampled[train_idx]
        y_val = y_train_resampled[val_idx]
        
        # Create the model with appropriate input shape
        model = create_model(X_train.shape[2], units=config['units'], activation=config['activation'],
                             dropout=config['dropout'], optimizer=config['optimizer'], layers=config['layers'])
        
        # Define checkpoint filepath for this fold
        filepath = os.path.join(output_dir, f"best_model_fold_{i+1}.keras")
        checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
        
        # Fit the model
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=10, batch_size=batch_size, callbacks=[checkpoint, early_stopping])
        
        # Evaluate predictions on validation set
        y_pred_val = model.predict(X_val).flatten()
        
        # Apply calibration
        prob_true, prob_pred = calibrate_predictions(y_val, y_pred_val)
        calibration_df = pd.DataFrame({'prob_true': prob_true, 'prob_pred': prob_pred})
        calibration_path = os.path.join(output_dir, f'calibration_fold_{i+1}.csv')
        calibration_df.to_csv(calibration_path, index=False)
        
        y_pred_train = model.predict(X_train)
        y_pred_train_binary = K.cast((y_pred_train > 0.5), 'int32')
        y_pred_val_binary = K.cast((y_pred_val > 0.5), 'int32')
        y_train_tensor = K.constant(y_train)
        y_val_tensor = K.constant(y_val)
        
        # Compute Matthews correlation using the imported function
        matthews_train = matthews(y_train_tensor, y_pred_train_binary)
        matthews_val = matthews(y_val_tensor, y_pred_val_binary)
        matthews_train_val = K.get_value(matthews_train)
        matthews_val_val = K.get_value(matthews_val)
        
        # Compute bootstrap metrics
        bootstrap_results = bootstrap_metrics(y_val, y_pred_val.ravel(), 1000)
        results_list = []
        for metric, values in bootstrap_results.items():
            mean_value = np.mean(values)
            ci_lower, ci_upper = np.percentile(values, [2.5, 97.5])
            results_list.append({ 
                'Fold': i+1,
                'Metric': metric,
                'Mean': mean_value,
                '95% CI Lower': ci_lower,
                '95% CI Upper': ci_upper
            })
        auc_records.append({
            'auc_train': roc_auc_score(y_train, y_pred_train),
            'auc_val': roc_auc_score(y_val, y_pred_val),
            'matthews_train': matthews_train_val,
            'matthews_val': matthews_val_val,
            'cv_num': i,
            **config
        })
        
        cv_record = {k: np.mean(v) for k, v in history.history.items()}
        cv_record['cv_num'] = i
        cv_record.update(config)
        cv_records.append(cv_record)
        
        val_accuracy = np.max(history.history['val_accuracy'])
        val_auc = roc_auc_score(y_val, y_pred_val)
        
        # Calculate hourly metrics
        hourly_auc, hourly_prc_auc = calculate_hourly_metrics(y_val, y_pred_val)
        for hour in range(1, X_val.shape[1] + 1):
            X_val_hour = X_val[:, :hour, :]
            y_pred_val_hour = model.predict(X_val_hour).flatten()
            auc_hour = roc_auc_score(y_val, y_pred_val_hour)
            accuracy_hour = accuracy_score(y_val, y_pred_val_hour > 0.5)
            precision_hour = precision_score(y_val, y_pred_val_hour > 0.5)
            recall_hour = recall_score(y_val, y_pred_val_hour > 0.5)
            mcc_hour = matthews_corrcoef(y_val, y_pred_val_hour > 0.5)
            hourly_metrics.append({
                'fold': i+1,
                'hour': hour,
                'auc': auc_hour,
                'accuracy': accuracy_hour,
                'precision': precision_hour,
                'recall': recall_hour,
                'mcc': mcc_hour
            })
        
        # Save hourly metrics after each fold
        hourly_metrics_df = pd.DataFrame(hourly_metrics)
        hourly_metrics_path = os.path.join(output_dir, f'hourly_metrics_fold_{i+1}.csv')
        hourly_metrics_df.to_csv(hourly_metrics_path, index=False)
        
        # Store calibrated metrics
        calibrated_auc = roc_auc_score(y_val, y_pred_val)
        calibrated_accuracy = accuracy_score(y_val, y_pred_val > 0.5)
        calibrated_precision = precision_score(y_val, y_pred_val > 0.5)
        calibrated_recall = recall_score(y_val, y_pred_val > 0.5)
        calibrated_mcc = matthews_corrcoef(y_val, y_pred_val > 0.5)
        calibrated_metrics.append({
            'fold': i+1,
            'config': config,
            'calibrated_auc': calibrated_auc,
            'calibrated_accuracy': calibrated_accuracy,
            'calibrated_precision': calibrated_precision,
            'calibrated_recall': calibrated_recall,
            'calibrated_mcc': calibrated_mcc
        })
        
        calibrated_val_preds.append(pd.DataFrame({
            'true_label': y_val,
            'predicted_prob': y_pred_val
        }))
        
        # Generate and save calibration plot
        disp = CalibrationDisplay.from_predictions(y_val, y_pred_val, n_bins=10, strategy='uniform')
        plot_path = os.path.join(output_dir, f'calibration_plot_fold_{i+1}.png')
        disp.figure_.savefig(plot_path)
        print(f"Calibration plot saved to {plot_path}")
        
        # Append per-fold results to CSV
        with open(results_filepath, 'a') as result_file:
            result_file.write(f"{i+1},{val_auc},{val_accuracy},{config}\n")
        print(f"Fold {i+1}, AUC: {val_auc}, Accuracy: {val_accuracy}, Config: {config}")

        # Update best model if current validation AUC is higher
        if val_auc > best_auc:
            best_auc = val_auc
            model.save(best_model_path)
            best_model_details = f"Best model so far: {config} with AUC: {val_auc} at fold {i+1}"
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(os.path.join(output_dir, 'training_history_best_model.csv'), index=False)
            hourly_results_df = pd.DataFrame({
                'true_label': y_val,
                'predicted_proba': y_pred_val
            })
            if not os.path.exists(results_calibration):
                hourly_results_df.to_csv(results_calibration, index=False)
            else:
                hourly_results_df.to_csv(results_calibration, mode='a', index=False)
    
print(best_model_details)

# Save overall results
AUChistory = pd.DataFrame(auc_records)
auc_history_path = os.path.join(output_dir, 'AUC_history_gridsearch.csv')
AUChistory.to_csv(auc_history_path, index=False, sep=',', header=True)

bootstrap_df.to_csv(bootstrap_results_path, index=False)

cv_df = pd.DataFrame(cv_records)
cv_df.to_csv(cv_results_path, index=False, sep=',', header=True)

hourly_results_df = pd.DataFrame(hourly_metrics)
hourly_results_path = os.path.join(output_dir, 'hourly_metrics.csv')
hourly_results_df.to_csv(hourly_results_path, index=False)

calibrated_metrics_df = pd.DataFrame(calibrated_metrics)
calibrated_metrics_path = os.path.join(output_dir, 'calibrated_metrics.csv')
calibrated_metrics_df.to_csv(calibrated_metrics_path, index=False)

# Save calibrated validation predictions
calibrated_val_preds_df = pd.concat(calibrated_val_preds, ignore_index=True)
calibrated_val_preds_path = os.path.join(output_dir, 'calibrated_val_predictions.csv')
calibrated_val_preds_df.to_csv(calibrated_val_preds_path, index=False)

