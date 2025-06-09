import numpy as np

def predict_sweetness(model, input_df):
    return model.predict(input_df)[0]

def classify_sweetness(value, data):
    percentiles = np.percentile(data['Sweetness'], [25, 75])
    if value < percentiles[0]:
        return "Less sweet"
    elif value < percentiles[1]:
        return "Moderately Sweet"
    return "Very sweet"