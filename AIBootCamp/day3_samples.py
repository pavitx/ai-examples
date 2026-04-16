import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Generate synthetic data
np.random.seed(42)

X = np.random.rand(100, 1) * 100
y = 3 * X**2 + 2 * X + np.random.randn(100, 1) * 5

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

y_pred = model.predict(X_poly)

plt.scatter(X, y, color='blue', label="Actual data")
plt.scatter(X, y_pred, color='red', label="Predicted data")
plt.title("Polynomial Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

mse = mean_squared_error(y, y_pred)
print("Mean Square Error (MSE)", mse)