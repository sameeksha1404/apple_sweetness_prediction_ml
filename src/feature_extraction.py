from skimage import io, color
import numpy as np
import pandas as pd

def extract_features(image_path, scaler):
    image = io.imread(image_path)
    fruit_ripeness = 3  # Placeholder
    fruit_r = np.mean(image[:, :, 0])
    fruit_g = np.mean(image[:, :, 1])
    fruit_b = np.mean(image[:, :, 2])

    df = pd.DataFrame([{
        'Ripeness': fruit_ripeness,
        'Color_R': fruit_r,
        'Color_G': fruit_g,
        'Color_B': fruit_b
    }])
    df = df[['Ripeness', 'Color_R', 'Color_G', 'Color_B']]
    df[df.columns] = scaler.transform(df[df.columns])
    return df