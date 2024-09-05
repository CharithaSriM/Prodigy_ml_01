import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    'SquareFootage': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    'Bedrooms': [3, 3, 4, 4, 5, 5, 4, 3, 3, 5],
    'Bathrooms': [2, 2, 3, 3, 4, 2, 2, 3, 2, 4],
    'Price': [300000, 320000, 340000, 360000, 400000, 450000, 470000, 390000, 380000, 500000]
}
df = pd.DataFrame(data)

# Prepare the data
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split the dataset into training and testing sets with more samples in the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

# R-squared score (check for enough samples)
if len(X_test) > 1:
    r2 = r2_score(y_test, y_pred)
else:
    r2 = "Not defined (less than 2 samples in the test set)"

# Output the coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Print results
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Coefficients: {coefficients}")
print(f"Intercept: {intercept}")