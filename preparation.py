

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import SMOTE

def tokenize_and_embed(group, embedding_dim=16):
    
    time_windows = group['hour'].unique()
    time_window_vectors = []
    for window in time_windows:
        # Drop non-feature columns; assume columns 'hour', 'patientid', and 'Outcome' are not features.
        window_data = group[group['hour'] == window].drop(columns=['hour', 'patientid', 'Outcome'])
        token_vector = window_data.mean(axis=0).values
        time_window_vectors.append(token_vector)
    return np.stack(time_window_vectors)

def process_data(csv_file, embedding_dim=16):

    df = pd.read_csv(csv_file)
    sequences = []
    labels = []
    for pid, group in df.groupby('patientid'):
        seq = tokenize_and_embed(group, embedding_dim=embedding_dim)
        sequences.append(seq)
        # Assume that all rows for a patient have the same 'Outcome'; take the first one.
        labels.append(group['Outcome'].iloc[0])
    return sequences, labels

def main():
    # --------------------------
    # Configuration – Update these values!
    # --------------------------
    train_data_file = ""   # Path to training CSV file
    test_data_file = ""     # Path to test CSV file
    
    # --------------------------
    # Process Training Data
    # --------------------------
    train_sequences, train_labels = process_data(train_data_file, embedding_dim=16)
    print(f"Processed {len(train_sequences)} training patients.")
    
    # Determine maximum sequence length from training data
    max_seq_len = max(len(seq) for seq in train_sequences)
    print(f"Maximum sequence length in training set: {max_seq_len}")
    
    # Pad training sequences so that all have length = max_seq_len
    padded_train = pad_sequences(train_sequences, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post')
    
    # --------------------------
    # Process Test Data
    # --------------------------
    test_sequences, test_labels = process_data(test_data_file, embedding_dim=16)
    print(f"Processed {len(test_sequences)} test patients.")
    # Pad test sequences using the same max_seq_len (for consistency)
    padded_test = pad_sequences(test_sequences, maxlen=max_seq_len, dtype='float32', padding='post', truncating='post')
    
    # --------------------------
    # Apply SMOTE to Training Data
    # --------------------------
    # Flatten training sequences from 3D to 2D
    X_train_flat = padded_train.reshape(padded_train.shape[0], -1)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, np.array(train_labels))
    # Determine feature dimension after flattening (should equal: max_seq_len * feature_dim)
    feature_dim = X_train_resampled.shape[1] // max_seq_len
    # Reshape the resampled training data back to 3D: (samples, max_seq_len, feature_dim)
    X_train_resampled = X_train_resampled.reshape(-1, max_seq_len, feature_dim)
    
    # --------------------------
    # Save Processed Datasets
    # --------------------------
    os.makedirs('data/token', exist_ok=True)
    np.save('data/token/X_resampled_new.npy', X_train_resampled)
    np.save('data/token/y_resampled_new.npy', y_train_resampled)
    np.save('data/token/X_test.npy', padded_test)
    np.save('data/token/y_test.npy', np.array(test_labels))
    
    print("Training and test datasets have been saved successfully.")

if __name__ == "__main__":
    main()
