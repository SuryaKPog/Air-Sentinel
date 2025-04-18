
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load and preprocess dataset
data = pd.read_csv("AirQuality.csv", delimiter=";")
data = data.drop(columns=['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'], errors='ignore')

# Encode categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Fill missing values
data = data.fillna(data.mean())

# Split features and target
X = data.drop(columns=['CO(GT)'])
y = data['CO(GT)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Print evaluation
print("Training R² Score:", train_r2)
print("Testing R² Score:", test_r2)
print("Training MSE:", train_mse)
print("Testing MSE:", test_mse)

# Plotting
plt.figure(figsize=(12, 5))

# R² Score Plot
plt.subplot(1, 2, 1)
plt.bar(["Train R²", "Test R²"], [train_r2, test_r2], color=["skyblue", "salmon"])
plt.title("R² Score Comparison")
plt.ylabel("R² Score")

# MSE Plot
plt.subplot(1, 2, 2)
plt.bar(["Train MSE", "Test MSE"], [train_mse, test_mse], color=["skyblue", "salmon"])
plt.title("Mean Squared Error Comparison")
plt.ylabel("MSE")

plt.tight_layout()
plt.show()

joblib.dump(model, "rf_air_quality.pkl")