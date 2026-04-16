import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

data = fetch_california_housing(as_frame=True)
df = data.frame

# select features: Median Income and Median House Value
X = df[['MedInc']]
y = df[['MedHouseVal']]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_predictions = lasso_model.predict(X_test)

ridge_MSE = mean_squared_error(y_test, ridge_predictions)
print("Ridge MSE: ", ridge_MSE)

lasso_MSE = mean_squared_error(y_test, lasso_predictions)
print("Lasso SE: ", lasso_MSE)

# visualize Ridge vs Lasso predictions
plt.figure(figsize=(10,6))
plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual data', alpha=0.5)
plt.scatter(X_test[:, 0], ridge_predictions, color='green', label='Ridge predictions', alpha=0.5)
plt.scatter(X_test[:, 0], lasso_predictions, color='orange', label='Lasso predictions', alpha=0.5)
plt.title('Ridge vs Lasso regression')
plt.xlabel('Median Income  (Transformed)')
plt.ylabel('Median House Value in California')
plt.legend()
plt.show()