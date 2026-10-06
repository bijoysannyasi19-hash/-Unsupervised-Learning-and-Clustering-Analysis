import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

def audit_data(df):
    """Performs a quick audit of the data."""
    audit_results = {
        'shape': df.shape,
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': int(df.duplicated().sum())
    }
    return audit_results

def preprocess_data(df):
    """
    Preprocesses the Wholesale Customers dataset for clustering.
    - Removes duplicates.
    - Separates 'Channel' and 'Region' as they are categorical labels.
    - Applies log1p transformation to highly skewed continuous features.
    - Applies StandardScaler.
    """
    df_clean = df.drop_duplicates().copy()
    
    # Separate labels
    labels = df_clean[['Channel', 'Region']].copy()
    features = df_clean.drop(['Channel', 'Region'], axis=1)
    
    # Store original scale features for profiling later
    original_features = features.copy()
    
    # Log transform to handle high right-skewness common in spending data
    features_log = np.log1p(features)
    
    # Scale features
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(features_log)
    features_scaled = pd.DataFrame(scaled_array, columns=features.columns, index=features.index)
    
    return original_features, features_scaled, labels

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data("data/wholesale_customers.csv")
    audit = audit_data(df)
    print("Audit:", audit)
    orig, scaled, labels = preprocess_data(df)
    print("Scaled Head:\n", scaled.head())
