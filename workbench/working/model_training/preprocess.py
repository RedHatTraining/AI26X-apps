"""
Data preprocessing utilities for server failure prediction.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_server_data(server_df):
    """
    Preprocess server metrics data for model inference.

    Args:
        server_df: DataFrame with server metrics (can be single row or multiple rows)

    Returns:
        DataFrame with preprocessed features ready for model prediction
    """
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(server_df, columns=["workload_type"])

    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)

    return X_scaled
