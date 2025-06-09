import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(data_path, feature_cols, target_col='Sweetness'):
    data = pd.read_csv(data_path)
    data = data[data[target_col] > 0]
    data[target_col] = data[target_col].fillna(data[target_col].mean())
    for col in feature_cols:
        data = data[pd.to_numeric(data[col], errors='coerce').notnull()]
    return data

def normalize_features(data, feature_cols):
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    return data, scaler