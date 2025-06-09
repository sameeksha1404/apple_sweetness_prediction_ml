import cv2
from src import preprocess, train_model, feature_extraction, predict
from utils.constants import FEATURES, DATA_PATH
import joblib

data = preprocess.load_and_clean_data(DATA_PATH, FEATURES)
data, scaler = preprocess.normalize_features(data, FEATURES)
model, X_test, y_test, cv_rmse = train_model.train_model(data, FEATURES, 'Sweetness')

# Save model and scaler
joblib.dump(model, 'model/linear_regression_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# Webcam capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('Capture - Press q to Snap', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('images/snapshot.jpg', frame)
        break
cap.release()
cv2.destroyAllWindows()

# Feature extraction
features_df = feature_extraction.extract_features('images/snapshot.jpg', scaler)
sweetness = predict.predict_sweetness(model, features_df)
category = predict.classify_sweetness(sweetness, data)

print(f"\nPredicted Sweetness: {sweetness:.2f}")
print(f"Sweetness Category: {category}")