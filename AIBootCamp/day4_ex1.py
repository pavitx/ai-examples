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
# np.arange(start, stop, step) returns a 1D array of evenly spaced values from start up to (but not including) stop, with the given step.
# It's like Python's range() but:
# works with floats (not just integers)
# returns a numpy array instead of a range object
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# predict probabilities for each point in the grid
# np.c_ is not a function — it's a indexing object that concatenates arrays column-wise (horizontally).
# xx.ravel() → flattens xx into a 1D array, e.g. shape (12100,)
# yy.ravel() → flattens yy into a 1D array, e.g. shape (12100,)
# np.c_[...] → stacks them side by side into a 2D array of shape (12100, 2)
# This is needed because model.predict() expects a 2D array where each row is one sample with all its features — exactly the same shape as X_train and X_test.
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# coolwarm: blue for class 0 (no purchase), red for class 1 (purchase)
# The result is the coloured background of the plot — it shows which region of the Age/Salary space the model classifies as "bought" vs "didn't buy".
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
# X_test['Age'] → x-coordinates of the 40 test points
# X_test['Salary'] → y-coordinates of the 40 test points
# c=y_test → colour each dot by its actual label (0 or 1), using coolwarm (blue/red)
# edgecolors='k' → black border around each dot so they stand out from the background
# If a dot's colour matches the background behind it → correct prediction. If it doesn't → misclassification.
plt.scatter(X_test['Age'], X_test['Salary'], c=y_test, edgecolors='k', cmap='coolwarm')
plt.title("Logistic regression decision boundary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()