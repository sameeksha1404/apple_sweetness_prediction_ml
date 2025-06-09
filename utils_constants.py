# utils/constants.py

# Thresholds for sweetness classification based on dataset percentiles
LESS_SWEET_THRESHOLD = 3.0
MORE_SWEET_THRESHOLD = 6.0

# Placeholder value for ripeness if not inferred from the image
DEFAULT_RIPENESS = 3

# Path settings
DATA_PATH = r'data/modified_apple-quality-with-rgb.csv'
SNAPSHOT_PATH = r'images/snapshot.jpg'
MODEL_PATH = r'model/linear_regression_model.pkl'
SCALER_PATH = r'model/scaler.pkl'

# Feature columns
FEATURE_COLUMNS = ['Ripeness', 'Color_R', 'Color_G', 'Color_B']
