import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# generate synthetic data
np.random.seed(42)

# Creates X as a (100, 1) column of random input values between 0 and 100 (uniformly distributed).
X = np.random.rand(100, 1) * 100
# Creates target values y using a linear rule y = 3X plus random Gaussian noise.
# np.random.randn(100, 1) * 2 adds noise with standard deviation about 2, so points are near a line but not perfectly on it (more realistic for regression).
y = 3 * X + np.random.randn(100, 1) * 2

X_Train, X_test, y_Train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_Train, y_Train)

y_pred = model.predict(X_test)

print("Slope: ", model.coef_[0][0])
print("Intercept: ", model.intercept_[0])

plt.scatter(X_test, y_test, color='blue', label="Actual")
plt.plot(X_test, y_pred, color='red', label="Predicted")
plt.title("Linear Regression Model")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE: ", mse)
print("R-squared: ", r2)
