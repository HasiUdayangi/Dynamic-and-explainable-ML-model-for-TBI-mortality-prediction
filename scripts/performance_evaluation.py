#!/usr/bin/env python3
"""
Performance Evaluation Script

This script:
  - Reads a CSV file (e.g., "performance_metrics.csv") containing performance metrics across folds.
  - Groups the data by hour and calculates the mean for each metric.
  - Plots AUC, Accuracy, Precision, Recall, and MCC over time.
  - Saves the resulting plot in a designated output folder.

Usage:
    python performance_evaluation.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Update the CSV file path if needed.
    metrics_csv = "performance_metrics.csv"
    
    if not os.path.exists(metrics_csv):
        print(f"Error: {metrics_csv} not found. Please ensure the file exists.")
        return
    
    # Read performance metrics from CSV
    df = pd.read_csv(metrics_csv)
    
    # Group by 'hour' and calculate mean metrics across folds
    mean_metrics = df.groupby('hour')[['auc', 'accuracy', 'precision', 'recall', 'mcc']].mean().reset_index()
    # Optionally, you can restrict to a certain time range (e.g., first 72 hours)
    # mean_metrics = mean_metrics[mean_metrics['hour'] <= 72]
    
    print("Average performance metrics by hour:")
    print(mean_metrics.head())
    
    # Define output folder for saving the plot
    output_folder = "performance_analysis/validation_performances"
    os.makedirs(output_folder, exist_ok=True)
    
    # Plot all metrics in one graph
    plt.figure(figsize=(12, 7))
    plt.plot(mean_metrics['hour'], mean_metrics['auc'], label='AUC', marker='o', linestyle='-')
    plt.plot(mean_metrics['hour'], mean_metrics['accuracy'], label='Accuracy', marker='s', linestyle='--')
    plt.plot(mean_metrics['hour'], mean_metrics['precision'], label='Precision', marker='^', linestyle='-.')
    plt.plot(mean_metrics['hour'], mean_metrics['recall'], label='Recall', marker='D', linestyle=':')
    plt.plot(mean_metrics['hour'], mean_metrics['mcc'], label='MCC', marker='x', linestyle='-')
    
    # Customize the plot
    plt.xlabel('Hours')
    plt.ylabel('Mean Performance Metrics')
    plt.title('Performance Metrics over Time (First 72 Hours)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to the output folder
    output_path = os.path.join(output_folder, 'combined_performance_metrics_.png')
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()

