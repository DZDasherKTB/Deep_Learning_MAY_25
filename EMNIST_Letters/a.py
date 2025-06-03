# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Fetch cryptocurrency data using yfinance (e.g., Bitcoin for the last 1 year)
crypto_symbol = "BTC-USD"  # You can change this to any other cryptocurrency (ETH-USD, etc.)
data = yf.download(crypto_symbol, start="2024-01-01", end="2025-01-01")

# Use 'Close' prices as the target (y) and 'Open' and 'High' as features (X)
X = data[['Open', 'High']].values  # Example features: Open and High prices
y = data['Close'].values  # Target: Close price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Apply a simple moving average (Smoothing)
window_size = 3  # Adjust the window size based on your data's noise level
smoothed_predictions = np.convolve(y_pred, np.ones(window_size) / window_size, mode='valid')

# Plot the Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', marker='o', color='blue')
plt.plot(y_pred, label='Predicted', marker='x', color='red')
plt.title('Actual vs Predicted Values')
plt.xlabel('Test Data Index')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plot the Smoothed Predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', marker='o', color='blue')
plt.plot(range(window_size-1, len(y_pred)), smoothed_predictions, label='Smoothed Predicted', marker='x', color='green')
plt.title('Actual vs Smoothed Predicted Values')
plt.xlabel('Test Data Index')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plot the Prediction Error (Actual - Predicted)
error = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.plot(error, label='Prediction Error', color='red', marker='o')
plt.title('Prediction Error (Actual - Predicted)')
plt.xlabel('Test Data Index')
plt.ylabel('Error')
plt.legend()
plt.show()
