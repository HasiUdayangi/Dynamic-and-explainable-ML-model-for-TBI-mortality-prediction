BUCKET_NAME = ""
# Input CSV files are assumed to be stored under this prefix in S3
INPUT_PREFIX = ""
# Processed files (data length fixed) will be stored here
FIXED_PREFIX = "/"
# Imputed files will be stored here
IMPUTED_PREFIX = ""

# Create an S3 client (ensure your AWS credentials are set up)
s3_client = boto3.client('s3')


# ==========================
# Functions
# ==========================
def process_and_upload_patient_file(patient_id, data_length):
    """
    Reads a patient CSV from S3, ensures that every hour from 0 to data_length is present,
    fills missing values with '0', and uploads the processed file back to S3.
    """
    csv_filename = f"s3://{BUCKET_NAME}/{INPUT_PREFIX}{patient_id}.csv"
    patient_data = pd.read_csv(csv_filename)
    
    # Retain only the required columns
    patient_data = patient_data[columns_to_keep]
    
    # Create a DataFrame with all hours (0 to data_length-1)
    all_hours = pd.DataFrame({'hour': list(range(data_length)), 'patientid': [patient_id] * data_length})
    # Merge to ensure every hour is present
    patient_data = pd.merge(all_hours, patient_data, on=['hour', 'patientid'], how='left')
    
    # Fill missing values with '0' (modify if necessary)
    patient_data.fillna('0', inplace=True)
    
    # Upload the processed CSV to S3 under FIXED_PREFIX
    output_path = f"s3://{BUCKET_NAME}/{FIXED_PREFIX}{patient_id}.csv"
    patient_data.to_csv(output_path, index=False)
    print(f"Processed file for {patient_id} saved to {output_path}")
    return patient_data


def imputation(patient_id):
    """
    Reads a processed patient CSV from S3 (under FIXED_PREFIX), replaces zeros with NaN
    in the specified columns, performs KNN imputation, and uploads the imputed file back to S3.
    """
    csv_filename = f"s3://{BUCKET_NAME}/{FIXED_PREFIX}{patient_id}.csv"
    patient_data = pd.read_csv(csv_filename)
    
    # Replace 0 with NaN in the columns to impute
    patient_data[columns_to_impute] = patient_data[columns_to_impute].replace('0', np.nan)
    
    # Perform KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    patient_data[columns_to_impute] = imputer.fit_transform(patient_data[columns_to_impute])
    
    # Upload the imputed CSV to S3 under IMPUTED_PREFIX
    output_path = f"s3://{BUCKET_NAME}/{IMPUTED_PREFIX}{patient_id}.csv"
    patient_data.to_csv(output_path, index=False)
    print(f"{patient_id} imputed and saved to {output_path}")
    return patient_data


def main():
    # Loop through each patient in df_ids and perform imputation if not skipped
    for index, row in df_ids.iterrows():
        patient_id = row['Patientid']
        if patient_id not in ids_to_skip:
            # If needed, first process and upload the fixed-length CSV file:
            # process_and_upload_patient_file(patient_id, data_length=24)  # Example: 24 hours of data
            # Then perform imputation:
            imputation(patient_id)


if __name__ == "__main__":
    main()
