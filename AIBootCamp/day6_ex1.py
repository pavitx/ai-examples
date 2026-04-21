# Day 6 Exercise 1 - K-Nearest Neighbors (KNN) Classification
#
# HOW KNN CLASSIFIES:
#   Idea: "you are what your neighbors are." No training phase - all work happens at prediction time.
#   At prediction time for each new point:
#     1. Calculate distance (Euclidean) from that point to every point in the training set
#     2. Pick the k closest training points (the "k nearest neighbors")
#     3. Take a majority vote among those k neighbors - the most common class wins
#   KNN is called a "lazy learner" because it defers all computation to prediction time.
#
# WHY StandardScaler IS IMPORTANT FOR KNN:
#   KNN is distance-based, so features with larger ranges dominate unfairly.
#   StandardScaler normalizes all features to mean=0, std=1 so they contribute equally.
#
# EFFECT OF k:
#   Small k (e.g. k=1) -> very sensitive to individual points, prone to overfitting/noise
#   Large k            -> smoother decision boundary, but may underfit
#   The loop tries k=1 to 10 to find the sweet spot for this dataset.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# experiment with different values of k

# accuracy is 100% for k = 1 through 10, which is quite fortunate for the 30 test samples
# Iris has 150 samples, with test_size=0.2 there are 30 test samples
# other values of random_state or a bigger test size might not give 100%
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    # fit() for KNN does NO computation - it simply stores the training data in memory.
    # All the actual work (distance calculation + voting) happens later in predict().
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"k = {k}, accuracy: {accuracy:.2f}")
