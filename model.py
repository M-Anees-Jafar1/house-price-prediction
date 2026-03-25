import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("housing_data.csv")

print(df.head())

# Check missing values
print(df.isnull().sum())

# Feature & target
X = df.drop("Price", axis=1)
y = df["Price"]

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# ================= USER INPUT PREDICTION =================

print("\n--- Enter House Details ---")

area = float(input("Enter Area (sq ft): "))
bedrooms = int(input("Enter Bedrooms: "))
bathrooms = int(input("Enter Bathrooms: "))
location = int(input("Enter Location Score (1-10): "))
age = int(input("Enter House Age: "))

# Convert input to array
input_data = np.array([[area, bedrooms, bathrooms, location, age]])

# Prediction
predicted_price = model.predict(input_data)

print("\n💰 Predicted House Price:", predicted_price[0])
# Plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
# ==========================================
# Portfolio Statement
# Developed an end-to-end house price prediction system 
# using linear regression with statistical evaluation, 
# feature scaling, and data visualization.
# ==========================================