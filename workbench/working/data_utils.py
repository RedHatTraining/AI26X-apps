"""
Data utilities for server failure prediction.
This module provides functions for data loading, preprocessing, and splitting.
"""

# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

import os
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_SERVER_METRICS_FILE = Path(__file__).parent / "server_metrics.csv"
DEFAULT_SCALER_FILE = Path(__file__).parent / "data_scaler.pkl"


def load_data(filepath: Path | str = DEFAULT_SERVER_METRICS_FILE):
    """Load the server metrics raw data."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    return df


def preprocess_features(
    df, scaler=None, save_scaler=True, scaler_path=DEFAULT_SCALER_FILE
):
    """
    Preprocess the features for model training or inference.

    Args:
        df: DataFrame with server metrics
        scaler: Optional pre-fitted StandardScaler for inference
        save_scaler: If True, save the fitted scaler to disk
        scaler_path: Path to save/load the scaler

    Returns:
        Tuple of (X_scaled, y, scaler) where:
            - X_scaled: Preprocessed features
            - y: Target variable (None if not in df)
            - scaler: Fitted StandardScaler
    """
    print("Preprocessing features...")

    # Check if target exists
    has_target = "failure_within_48h" in df.columns

    # Separate features and target
    if has_target:
        X = df.drop(["failure_within_48h", "server_id", "timestamp"], axis=1)
        y = df["failure_within_48h"]
    else:
        # For inference, these columns may not exist
        cols_to_drop = [col for col in ["server_id", "timestamp"] if col in df.columns]
        X = df.drop(cols_to_drop, axis=1)
        y = None

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=["workload_type"])

    # Scale features
    if scaler is None:
        # Training mode: fit new scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        if save_scaler:
            print(f"Saving scaler to {scaler_path}...")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
    else:
        # Inference mode: use provided scaler
        X_scaled = scaler.transform(X_encoded)

    X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)

    print(f"Preprocessed features shape: {X_scaled.shape}")
    return X_scaled, y


def load_scaler(scaler_path=DEFAULT_SCALER_FILE):
    """Load a saved StandardScaler from disk."""
    print(f"Loading scaler from {scaler_path}...")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets."""
    print(
        f"Splitting data: {int((1 - test_size) * 100)}% train, {int(test_size * 100)}% test..."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


def calculate_class_weights(y_train):
    """
    Calculate class weights to handle imbalanced datasets.

    The weights are inversely proportional to class frequencies.
    This helps the model pay more attention to the minority class.

    Args:
        y_train: Training labels (pandas Series or numpy array)

    Returns:
        Dictionary mapping class labels to weights
    """
    # Count samples per class
    n_samples = len(y_train)
    n_classes = len(y_train.unique())

    # Calculate weight for each class
    class_weights = {}
    for class_label in sorted(y_train.unique()):
        n_class_samples = (y_train == class_label).sum()
        weight = n_samples / (n_classes * n_class_samples)
        class_weights[class_label] = weight

    print("Class weights calculated:")
    for class_label, weight in class_weights.items():
        class_name = "No failure" if class_label == 0 else "Failure"
        print(f"  Class {class_label} ({class_name}): {weight:.4f}")

    return class_weights


def save_prepared_data(X_train, X_test, y_train, y_test):
    """Save the prepared datasets to CSV files."""
    print("Saving prepared datasets...")

    # Combine features and target for saving
    train_df = X_train.copy()
    train_df["failure_within_48h"] = y_train.values

    test_df = X_test.copy()
    test_df["failure_within_48h"] = y_test.values

    train_df.to_csv("train_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)

    print(f"Saved train_data.csv ({len(train_df)} samples)")
    print(f"Saved test_data.csv ({len(test_df)} samples)")


def prepare_data(input_file=DEFAULT_SERVER_METRICS_FILE):
    """
    Complete data preparation pipeline.

    Loads raw data, preprocesses it, splits into train/test,
    and saves everything including the scaler.
    """
    print("=" * 60)
    print("Server Failure Prediction - Data Preparation")
    print("=" * 60)

    # Load data
    df = load_data(input_file)

    # Preprocess features (fit and save scaler)
    X, y, scaler = preprocess_features(df, save_scaler=True)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save prepared data
    save_prepared_data(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("Files created:")
    print("  - train_data.csv")
    print("  - test_data.csv")
    print("  - scaler.pkl")
    print("=" * 60)

    return X_train, X_test, y_train, y_test, scaler


def preprocess_for_inference(df, scaler):
    """
    Preprocess data for model inference using saved scaler.

    Args:
        df: DataFrame with server metrics (without target variable)
        scaler_path: Path to saved scaler file

    Returns:
        Preprocessed features ready for prediction
    """
    X_scaled, _, _ = preprocess_features(df, save_scaler=False)
    return X_scaled


if __name__ == "__main__":
    # Run the complete data preparation pipeline
    prepare_data()
