"""
Data preparation script for server failure prediction.
This script loads, cleans, and prepares the server metrics data for model training.

INTENTIONAL BUG: This script contains a data leakage bug that students must find and fix.
"""

# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_SERVER_METRICS_FILE = Path(__file__).parent / "server_metrics.csv"


def load_data(filepath=DEFAULT_SERVER_METRICS_FILE):
    """Load the server metrics raw data."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    return df


def preprocess_features(df):
    """
    Preprocess the features for model training.

    BUG WARNING: This function contains a bug!
    Students need to identify and fix it using the debugger.
    """
    print("Preprocessing features...")

    # Separate features and target
    X = df.drop(["failure_within_48h", "server_id", "timestamp"], axis=1)
    y = df["failure_within_48h"]

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=["workload_type"])

    # BUG: Using wrong variable! Should use X_encoded, not X
    # X still contains the categorical 'workload_type' column
    # StandardScaler cannot handle string values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)

    print(f"Preprocessed features shape: {X_scaled.shape}")
    return X_scaled, y


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


def main():
    """Main data preparation pipeline."""
    print("=" * 60)
    print("Server Failure Prediction - Data Preparation")
    print("=" * 60)

    # Load data
    df = load_data()

    # Preprocess features (contains bug!)
    X, y = preprocess_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save prepared data
    save_prepared_data(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
