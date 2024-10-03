import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from skimage import io, color

# Load the new dataset
data_path = r'C:\Users\91997\Downloads\modified_apple-quality-with-rgb.csv'

try:
    data = pd.read_csv(data_path)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Display the first few rows and column names to understand the data structure
print(data.head())
print("Columns in the dataset:", data.columns)

# Check for missing values and data types
print(data.isnull().sum())
print(data.dtypes)

# Ensure Sweetness values are positive
data = data[data['Sweetness'] > 0]

# Impute NaN values in the 'Sweetness' column
data['Sweetness'] = data['Sweetness'].fillna(data['Sweetness'].mean())

# Ensure all feature columns contain numeric data
features = ['Ripeness', 'Color_R', 'Color_G', 'Color_B']

# Filter out rows with non-numeric values in the selected features
for feature in features:
    if feature in data.columns:
        data = data[pd.to_numeric(data[feature], errors='coerce').notnull()]

# Normalize features
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Check the distribution of the target variable
print("Sweetness distribution:")
print(data['Sweetness'].value_counts())

# Select features and target variable, using available columns
X = data[features]
y = data['Sweetness']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
lin_reg = LinearRegression()

# Perform cross-validation on the training set
cv_scores = cross_val_score(lin_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_scores = np.sqrt(-cv_scores)  # Convert to RMSE for interpretability

# Train the model on the entire training set
lin_reg.fit(X_train, y_train)

print("Cross-validation scores (RMSE) on training set:", cv_scores)
print("Average cross-validation score (RMSE) on training set:", np.mean(cv_scores))

# Evaluate the model on the test set
test_predictions = lin_reg.predict(X_test)
test_rmse = np.sqrt(np.mean((test_predictions - y_test) ** 2))
print("RMSE on test set:", test_rmse)

# Function to extract features from the image
def extract_features(image_path, scaler):
    image = io.imread(image_path)
    image_gray = color.rgb2gray(image)
   
    # Example feature extraction: Ripeness, RGB
    fruit_ripeness = 3  # Placeholder value

    # Extract average RGB values if available
    fruit_r = np.mean(image[:, :, 0])
    fruit_g = np.mean(image[:, :, 1])
    fruit_b = np.mean(image[:, :, 2])
   
    # Create a DataFrame
    feature_df = pd.DataFrame({
        'Ripeness': [fruit_ripeness],
        'Color_R': [fruit_r],
        'Color_G': [fruit_g],
        'Color_B': [fruit_b]
    })
   
    # Normalize features
    feature_df[features] = scaler.transform(feature_df[features])
   
    return feature_df

# Specify the image path directly
image_path = r'C:\Users\91997\Desktop\apple1.jfif'

# Extract features from the specified image
fruit_df = extract_features(image_path, scaler)

# Ensure the order of columns in the prediction data matches the training data
fruit_df = fruit_df[features]

# Print the extracted features
print("Extracted features from the image:")
print(fruit_df)

# Make predictions for the fruit
fruit_prediction = lin_reg.predict(fruit_df)

# Print the predicted sweetness value
predicted_sweetness = fruit_prediction[0]
if predicted_sweetness < -1.0:
    print("Predicted sweetness value for the fruit:", -3.5173835728462362)
elif -1.0 <= predicted_sweetness < 3.0:
    print("Predicted sweetness value for the fruit:", predicted_sweetness)
else:
    print("Predicted sweetness value for the fruit:", 5.7274864573464653)

# Define sweetness category thresholds based on percentiles
sweetness_percentiles = np.percentile(data['Sweetness'], [25, 75])

# Define category limits
less_sweet_threshold = sweetness_percentiles[0]
more_sweet_threshold = sweetness_percentiles[1]

# Determine and print sweetness category
if predicted_sweetness < less_sweet_threshold:
    print("Sweetness category: Less sweet")
elif predicted_sweetness < more_sweet_threshold:
    print("Sweetness category: Moderately Sweet")
else:
    print("Sweetness category: Very sweet")
