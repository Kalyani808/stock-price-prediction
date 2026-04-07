# Fix matplotlib error (no GUI needed)
import matplotlib
matplotlib.use('Agg')

# Import libraries
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# STEP 1: Download stock data
data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# STEP 2: Select features (better model)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# STEP 3: Create prediction column (30 days ahead)
future_days = 30
data['Prediction'] = data['Close'].shift(-future_days)

# STEP 4: Prepare data
X = np.array(data.drop(['Prediction'], axis=1))[:-future_days]
y = np.array(data['Prediction'])[:-future_days]

# STEP 5: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# STEP 6: Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 7: Predict
predictions = model.predict(X_test)

# STEP 8: Accuracy
score = r2_score(y_test, predictions)
print("Model Accuracy (R2 Score):", score)

# STEP 9: Plot prediction vs actual
plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.savefig("prediction_plot.png")

print("Prediction graph saved successfully!")

# STEP 10: Predict future prices
last_data = data.drop(['Prediction'], axis=1).tail(future_days).values

future_predictions = model.predict(last_data)

print("\nFuture 30 Days Prediction:")
print(future_predictions)





