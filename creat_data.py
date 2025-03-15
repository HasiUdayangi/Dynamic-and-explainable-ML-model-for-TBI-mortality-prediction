import os
import pickle
import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --------------------------
# Configuration – Update these values!
# --------------------------
BUCKET_NAME = ""
# Folder where imputed CSV files are stored
IMPUTED_PREFIX = ""
# Local temporary directory (optional)
LOCAL_TEMP_DIR = ""
# Output directory for the pickle files
OUTPUT_DIR = ""

# Create local temp and output directories if they do not exist
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize S3 client
s3_client = boto3.client('s3')

def list_imputed_csv_files(bucket, prefix):
    """List all CSV file keys in the given S3 bucket under the specified prefix."""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    files = response.get('Contents', [])
    csv_keys = [file['Key'] for file in files if file['Key'].endswith('.csv')]
    return csv_keys

def download_csv_from_s3(bucket, key, local_dir):
    """Download a CSV file from S3 to the specified local directory and return the local file path."""
    local_filename = os.path.join(local_dir, os.path.basename(key))
    s3_client.download_file(Bucket=bucket, Key=key, Filename=local_filename)
    return local_filename

def load_patient_data(bucket, prefix, local_dir):
    """
    Download and read all imputed CSV files from S3 under the given prefix.
    Returns a list of patient DataFrames.
    """
    csv_keys = list_imputed_csv_files(bucket, prefix)
    print(f"Found {len(csv_keys)} imputed CSV files in {bucket}/{prefix}.")
    patient_data_list = []
    for key in csv_keys:
        local_file = download_csv_from_s3(bucket, key, local_dir)
        try:
            df = pd.read_csv(local_file)
            # Ensure a 'patientid' column exists; if not, use the filename (without extension) as the patient id.
            if 'patientid' not in df.columns:
                df['patientid'] = os.path.splitext(os.path.basename(key))[0]
            patient_data_list.append(df)
        except Exception as e:
            print(f"Error reading {local_file}: {e}")
    return patient_data_list

def create_train_test_splits(patient_data_list, test_size=0.2, random_state=42):
    """
    Split the list of patient DataFrames into training and testing sets at the patient level.
    Returns:
      - X_train: List of DataFrames for training.
      - X_test: List of DataFrames for testing.
      - If a 'mortality' column exists, also extract labels (y_train and y_test).
    """
    # Extract patient IDs from the DataFrames
    patient_ids = [df['patientid'].iloc[0] for df in patient_data_list]
    # Create a DataFrame to facilitate splitting
    df_patients = pd.DataFrame({'patientid': patient_ids})
    
    # Split patient IDs into training and testing sets
    train_ids, test_ids = train_test_split(df_patients['patientid'], test_size=test_size, random_state=random_state)
    
    X_train = [df for df in patient_data_list if df['patientid'].iloc[0] in train_ids.values]
    X_test = [df for df in patient_data_list if df['patientid'].iloc[0] in test_ids.values]
    
    # If the files contain a target column 'mortality', extract labels.
    if patient_data_list and 'mortality' in patient_data_list[0].columns:
        y_train = [df['mortality'].iloc[0] for df in X_train]
        y_test = [df['mortality'].iloc[0] for df in X_test]
        return X_train, y_train, X_test, y_test
    else:
        return X_train, None, X_test, None

def save_splits(X_train, y_train, X_test, y_test, output_dir):
    """
    Save the training and testing sets as pickle files.
    """
    with open(os.path.join(output_dir, "X_train.pkl"), "wb") as f:
        pickle.dump(X_train, f)
    with open(os.path.join(output_dir, "X_test.pkl"), "wb") as f:
        pickle.dump(X_test, f)
    if y_train is not None and y_test is not None:
        with open(os.path.join(output_dir, "y_train.pkl"), "wb") as f:
            pickle.dump(y_train, f)
        with open(os.path.join(output_dir, "y_test.pkl"), "wb") as f:
            pickle.dump(y_test, f)
    print(f"Train/test splits saved to '{output_dir}'.")

def main():
    # 1. Load patient data from S3 imputed files
    patient_data_list = load_patient_data(BUCKET_NAME, IMPUTED_PREFIX, LOCAL_TEMP_DIR)
    
    if not patient_data_list:
        print("No patient data found in S3.")
        return
    
    # 2. Create train/test splits at the patient level
    X_train, y_train, X_test, y_test = create_train_test_splits(patient_data_list, test_size=0.2)
    print(f"Training set: {len(X_train)} patients, Testing set: {len(X_test)} patients")
    
    # 3. Save the splits as pickle files
    save_splits(X_train, y_train, X_test, y_test, OUTPUT_DIR)

if __name__ == "__main__":
    main()
