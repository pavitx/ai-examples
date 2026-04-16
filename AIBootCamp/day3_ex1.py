import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = fetch_california_housing(as_frame=True)
df = data.frame

# select features: Median Income and Median House Value
X = df[['MedInc']]
y = df[['MedHouseVal']]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)


y_pred = model.predict(X_poly)

plt.figure(figsize=(10,6))
plt.scatter(X,y, color='blue', label='Actual data', alpha=0.5)
plt.scatter(X,y_pred, color='red', label='Predicted curve', alpha=0.5)
plt.title('Polynomial Regression')
plt.xlabel('Median Income in California')
plt.ylabel('Median House Value in California')
plt.legend()
plt.show()

mse = mean_squared_error(y, y_pred)
print("Mean square error (MSE)", mse)