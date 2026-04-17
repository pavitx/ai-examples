import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

# np.random.rand(200, 2) generates a 200×2 matrix of random floats between 0 and 1.
# Multiplied by 10, values become 0–10.
# Column 0 = "Age", Column 1 = "Salary" (both just random numbers in [0, 10]).
np.random.seed(42)
n_samples = 200
X = np.random.rand(n_samples, 2) * 10
# X[:, 0] * 1.5 + X[:, 1] is a weighted combination of Age and Salary.
# If that sum > 15, the person bought (y=1), otherwise didn't (y=0).
# .astype(int) converts True/False booleans to 1/0.
# This is the ground truth decision boundary: a known linear rule so we can test how well logistic regression recovers it.
y = (X[:, 0] * 1.5 + X[:, 1] > 15).astype(int)

# Puts X into a DataFrame with named columns Age and Salary.
# Adds the target column Purchase (0 or 1).
df = pd.DataFrame(X, columns=['Age', 'Salary'])
df['Purchase'] = y

X_train, X_test, y_train, y_test = train_test_split(df[['Age', 'Salary']], df['Purchase'], test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("\n Classification report: ", classification_report(y_test, y_pred))

# plot decision boundary

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# predict probabilities for each point in the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X_test['Age'], X_test['Salary'], c=y_test, edgecolors='k', cmap='coolwarm')
plt.title("Logistic regression decision boundary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()