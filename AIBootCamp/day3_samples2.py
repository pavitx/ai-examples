import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(42)

X = np.random.rand(100, 1) * 100
y = 3 * X**2 + 2 * X + np.random.randn(100, 1) * 5

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)

lasso_model = Lasso(alpha=1)
lasso_model.fit(X_train, y_train)
lasso_predictions = lasso_model.predict(X_test)

ridge_MSE = mean_squared_error(y_test, ridge_predictions)
print("Ridge MSE: ", ridge_MSE)

lasso_MSE = mean_squared_error(y_test, lasso_predictions)
print("Lasso MSE: ", lasso_MSE)