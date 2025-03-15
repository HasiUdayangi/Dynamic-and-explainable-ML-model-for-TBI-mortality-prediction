#!/usr/bin/env python3
"""
Tokenize, Embed, and Apply SMOTE on Patient Vital Data

This script performs the following steps:
1. Reads a combined CSV file (containing patient vital data) with columns including:
   - 'patientid'
   - 'hour'
   - 'Outcome'
   - Other vital sign feature columns
2. For each patient, tokenizes and embeds each time window by averaging the features.
3. Creates sequences and labels for each patient.
4. Pads the sequences to a uniform length.
5. Splits the dataset into training (80%) and testing (20%) sets.
6. Flattens the training sequences, applies SMOTE for class balancing,
   and reshapes the result back to 3D for LSTM input.
7. Saves the resulting datasets as .npy files.
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def tokenize_and_embed(group, embedding_dim=16):
    """
    Tokenizes and embeds each time window for a given patient group.
    The function averages the features within each unique 'hour' window.
    
    Parameters:
        group (pd.DataFrame): DataFrame for a single patient.
        embedding_dim (int): Embedding dimension (not used directly, as we average features).
    
    Returns:
        np.ndarray: 2D array where each row is the averaged feature vector for a given hour.
    """
    # Get unique hours in the group
    time_windows = group['hour'].unique()
    time_window_vectors = []
    for window in time_windows:
        # Drop the columns that are not features
        window_data = group[group['hour'] == window].drop(columns=['hour', 'patientid', 'Outcome'])
        token_vector = window_data.mean(axis=0).values  # Average the feature vectors
        time_window_vectors.append(token_vector)
    return np.stack(time_window_vectors)

def main():
    # File containing the combined vital data (update with your actual file path)
    combined_data_file = "combined_data.csv"
    combined_data = pd.read_csv(combined_data_file)
    
    # Create sequences and labels for each patient
    sequences = []
    labels = []
    embedding_dim = 16  # Set embedding dimension (for consistency)
    
    # Group the data by patientid
    for pid, group in combined_data.groupby('patientid'):
        patient_sequence = tokenize_and_embed(group, embedding_dim=embedding_dim)
        sequences.append(patient_sequence)
        labels.append(group['Outcome'].iloc[0])
    
    # Convert sequences and labels to arrays
    sequences = np.array(sequences, dtype=object)  # Use object dtype for variable-length sequences
    labels = np.array(labels)
    
    # Determine maximum sequence length across patients
    max_seq_len = max(len(seq) for seq in sequences)
    
    # Pad sequences to ensure all have the same length
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post')
    
    # Split the dataset into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Flatten the training set for SMOTE (convert 3D to 2D)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Not used for SMOTE; kept for consistency
    
    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)
    
    # Reshape the resampled training data back to 3D: (samples, max_seq_len, feature_dim)
    feature_dim = X_train_resampled.shape[1] // max_seq_len
    X_train_resampled = X_train_resampled.reshape(-1, max_seq_len, feature_dim)
    
    # Save the processed training and testing sets
    np.save('data/token/X_resampled_new.npy', X_train_resampled)
    np.save('data/token/y_resampled_new.npy', y_train_resampled)
    np.save('data/token/X_test.npy', X_test)
    np.save('data/token/y_test.npy', y_test)
    
    print("Training and testing datasets have been saved successfully.")

if __name__ == "__main__":
    main()

