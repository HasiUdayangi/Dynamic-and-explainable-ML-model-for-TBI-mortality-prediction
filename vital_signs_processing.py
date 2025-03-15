#!/usr/bin/env python3
"""
Vital Signs Processing Module

This module contains functions to:
  - Transform vital signs data from long to wide format.
  - Calculate hourly statistics (min, max, median) for various vital signs.
  
Functions:
    prepare_vital_signs_pivot(df, admission_time)
        - Pivots the DataFrame, renames parameter columns, calculates time since admission,
          and inserts a patient ID column.
        
    calculate_hourly_vitals(df)
        - Groups the pivoted data by patient and hour to compute min, max, and median for vital signs.
        
Usage:
    This module expects a long-format CSV file (e.g., "vitals_long.csv") with columns:
        - 'time'
        - 'value'
        - 'parameterid'
    You must also provide an admission time and patient_id.
    
    The main block demonstrates the usage and saves the hourly statistics to a CSV file.
"""

from IPython.display import HTML, clear_output
import sys
sys.path.append("/home/ec2-user/SageMaker/")
sys.path.append("/home/ec2-user/SageMaker/IHQ-DataTrustPlatform/")
sys.path.append("/home/ec2-user/SageMaker/relationship between HRV and others/Vital sign feature/")

import boto3
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime, timedelta, date

# Import custom modules (ensure these paths are correct in your repo)
from utils import Athena_Query, s3, LabelStore
from utils.sql_query import SqlQuery

# Initialize Athena Query instance if needed (not used directly here)
athena = Athena_Query()

# Vital Sign Extractor and Imputation Processor imports
from vital_sign_extraction import VitalSignExtractor
from utils.waveform_viewer2 import Waveform_Extract, Waveform_Chart
vital = VitalSignExtractor()

from vital_imputation_processor import HealthDataProcessor
imputor = HealthDataProcessor()

# Parameter IDs for vital signs
parameter_ids = [3885,   # Heart Rate
                 3888,   # Diastolic Blood Pressure
                 3887,   # Systolic Blood Pressure
                 5436,   # ABP Mean
                 3951,   # SpO2
                 3976,   # Respiratory rate
                 3910,   # Temperature
                 4083]   # Intra-Cranial Pressure

def get_vitalsign_query(patient_ids, parameter_ids):
    patient_ids_string = "(" + ",".join([f"'{id}'" for id in patient_ids]) + ")"
    parameter_ids_string = "(" + ",".join([str(id) for id in parameter_ids]) + ")"
    
    query = f"""
    SELECT time, value, parameterid
    FROM metavision_deid_dtm.signals
    WHERE parameterid IN {parameter_ids_string} AND patientid IN {patient_ids_string}
    """
    return query

def extract_vital_signs(patient_ids, parameter_ids, admission_date):
    """
    Extracts vital signs data from Athena for given patient(s) and calculates time since admission.
    """
    query = get_vitalsign_query(patient_ids, parameter_ids)
    df = athena.query_as_pandas(query).drop_duplicates().reset_index(drop=True)
    df.sort_values('time', inplace=True)
    df['time_since_admission'] = (pd.to_datetime(df['time']) - datetime.strptime(admission_date, "%Y-%m-%d %H:%M:%S.%f")).dt.total_seconds() / 3600.0
    return df

def prepare_vital_signs_pivot(df, admission_time):
    """
    Transforms a DataFrame from long to wide format, assigns descriptive names to the parameters,
    drops the 'parameterid' level, calculates the time since admission for each entry, and resets 'time' as a regular column.
    """
    # Convert 'time' and 'admission_time' to datetime
    df['time'] = pd.to_datetime(df['time'])
    admission_time = pd.to_datetime(admission_time)

    # Create a pivot table with 'time' as index and parameter IDs as columns
    pivot_df = df.pivot_table(index='time', columns='parameterid', values='value', aggfunc='mean')

    # Mapping of parameter IDs to descriptive names
    column_names = {
        3885: 'HR',
        3888: 'Diastolic BP',
        3887: 'Systolic BP',
        5436: 'ABP Mean',
        3951: 'SpO2',
        3910: 'Temperature',
        4083: 'Intra-Cranial Pressure',
        3976: 'Respiratory rate'
    }

    pivot_df = pivot_df.rename(columns=column_names)
    pivot_df.columns.name = None  # Remove the column index name

    # Reset index to turn 'time' into a regular column
    pivot_df = pivot_df.reset_index()

    # Calculate time since admission (in hours)
    pivot_df['time_since_start'] = (pivot_df['time'] - admission_time).dt.total_seconds() / 3600
    # Insert patient_id as the first column (assumed to be a global variable)
    pivot_df.insert(0, 'patientid', patient_id)

    return pivot_df

def calculate_hourly_vitals(df):
    """
    Calculates hourly statistics (min, max, median) for vital signs and returns a reformatted DataFrame.
    """
    # Round down 'time_since_start' to the nearest hour
    df['hour'] = df['time_since_start'].apply(lambda x: int(x))
    
    # Group by patientid and hour, then aggregate
    grouped = df.groupby(['patientid', 'hour'])
    hourly_stats = grouped.agg({
        'HR': ['min', 'max', 'median'],
        'Diastolic BP': ['min', 'max', 'median'],
        'Systolic BP': ['min', 'max', 'median'],
        'ABP Mean': ['min', 'max', 'median'],
        'SpO2': ['min', 'max', 'median'],
        'Temperature': ['min', 'max', 'median'],
        'Intra-Cranial Pressure': ['min', 'max', 'median'],
        'Respiratory rate': ['min', 'max', 'median']
    }).reset_index()

    # Flatten MultiIndex columns
    hourly_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in hourly_stats.columns.values]

    # Rename columns for clarity
    hourly_stats.rename(columns={
        'HR_min': 'HR_min',
        'HR_max': 'HR_max',
        'HR_median': 'HR_median',
        'Diastolic BP_min': 'DBP_min',
        'Diastolic BP_max': 'DBP_max',
        'Diastolic BP_median': 'DBP_median',
        'Systolic BP_min': 'SBP_min',
        'Systolic BP_max': 'SBP_max',
        'Systolic BP_median': 'SBP_median',
        'ABP Mean_min': 'ABP_mean_min',
        'ABP Mean_max': 'ABP_mean_max',
        'ABP Mean_median': 'ABP_mean_median',
        'SpO2_min': 'SpO2_min',
        'SpO2_max': 'SpO2_max',
        'SpO2_median': 'SpO2_median',
        'Temperature_min': 'Temperature_min',
        'Temperature_max': 'Temperature_max',
        'Temperature_median': 'Temperature_median',
        'Intra-Cranial Pressure_min': 'ICP_min',
        'Intra-Cranial Pressure_max': 'ICP_max',
        'Intra-Cranial Pressure_median': 'ICP_median',
        'Respiratory rate_min': 'RR_min',
        'Respiratory rate_max': 'RR_max',
        'Respiratory rate_median': 'RR_median'
    }, inplace=True)

    return hourly_stats

# Global variables for this demonstration (update with actual values if needed)
admission_time = '2014-02-15 13:00:00.000'
patient_id = 'CEC39FCE110F42D170901F6CBA7FC3BC'

# Main execution
if __name__ == "__main__":
    # Read long-format vital signs data; update filename if necessary
    df = pd.read_csv("vitals_long.csv")
    
    # Prepare pivoted vital signs DataFrame
    n_df = prepare_vital_signs_pivot(df, admission_time)
    
    # Calculate hourly vital statistics
    hourly_stats = calculate_hourly_vitals(n_df)
    
    # Save the hourly statistics to a CSV file locally (you can later push it to S3)
    output_filename = "B001C98394B727A21644C9220D967092.csv"
    hourly_stats.to_csv(output_filename, index=False)
    print(f"Hourly vital statistics saved to '{output_filename}'."

