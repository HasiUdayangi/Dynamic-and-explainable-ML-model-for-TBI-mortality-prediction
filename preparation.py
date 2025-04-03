

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.over_sampling import SMOTE
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

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
    
def dtw_smote(X_train, y_train, minority_class=1, N=100, k=5):
    # Extract minority samples
    minority_idx = np.where(y_train == minority_class)[0]
    X_min = X_train[minority_idx]
    n_min = X_min.shape[0]
    
    # Calculate the number of synthetic samples to generate
    n_synthetic = int((N / 100.0) * n_min)
    synthetic_samples = []
    
    # Initialize DTW-based neighbor search using tslearn's KNeighborsTimeSeriesClassifier
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=k, metric="dtw")
    dummy_labels = np.full(n_min, minority_class)
    knn.fit(X_min, dummy_labels)
    
    for _ in range(n_synthetic):
        i = np.random.randint(0, n_min)
        sample = X_min[i]
        
        # Find k DTW neighbors (excluding the sample itself)
        nn_indices = knn.kneighbors([sample], return_distance=False)[0]
        nn_indices = nn_indices[nn_indices != i]
        if len(nn_indices) == 0:
            continue  # Skip if no neighbor is found
        j = np.random.choice(nn_indices)
        neighbor = X_min[j]
        
        # Generate a synthetic sample via interpolation
        gap = np.random.rand()  # Random factor between 0 and 1
        synthetic_sample = sample + gap * (neighbor - sample)
        synthetic_samples.append(synthetic_sample)
    
    if synthetic_samples:
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.full(len(synthetic_samples), minority_class)
    else:
        X_synthetic = np.empty((0, X_train.shape[1], X_train.shape[2]))
        y_synthetic = np.array([])
    
    return X_synthetic, y_synthetic
    
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

    # # Flatten training sequences from 3D to 2D
    # X_train_flat = padded_train.reshape(padded_train.shape[0], -1)
    # smote = SMOTE(random_state=42)
    # X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, np.array(train_labels))
    # # Determine feature dimension after flattening (should equal: max_seq_len * feature_dim)
    # feature_dim = X_train_resampled.shape[1] // max_seq_len
    # # Reshape the resampled training data back to 3D: (samples, max_seq_len, feature_dim)
    # X_train_resampled = X_train_resampled.reshape(-1, max_seq_len, feature_dim)

    train_labels_arr = np.array(train_labels)
    # Apply DTW-SMOTE on the padded training data
    X_train_synthetic, y_train_synthetic = dtw_smote(padded_train, train_labels_arr, minority_class=1, N=100, k=5)
    
    # Combine original training data with synthetic samples
    X_train_resampled = np.concatenate((padded_train, X_train_synthetic), axis=0)
    y_train_resampled = np.concatenate((train_labels_arr, y_train_synthetic), axis=0)
    
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
