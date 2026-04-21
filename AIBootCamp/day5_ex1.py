# Day 5 Exercise 1 - Random Forest Classification with Cross-Validation
#
# IRIS DATASET:
#   Classic tabular dataset (NOT images). 150 samples of iris flowers, 3 species:
#     - Iris setosa, Iris versicolor, Iris virginica
#   Each sample has 4 numerical features (measurements in cm):
#     sepal length, sepal width, petal length, petal width
#   data.data  -> 150x4 feature matrix (X)
#   data.target -> 150 labels (0, 1, 2 for each species)
#
# RANDOM FOREST CLASSIFIER:
#   Ensemble method: builds many decision trees and combines their votes.
#   - Decision tree: splits data with yes/no questions on features (e.g. petal length < 2.5?)
#   - Random Forest fixes overfitting via:
#       * Bagging: each tree trains on a random subset of data (with replacement)
#       * Random feature selection: at each split, only a random subset of features is considered
#   - Final prediction = majority vote across all trees (default: 100 trees)
#   - random_state=42 fixes randomness for reproducibility
#
# K-FOLD CROSS-VALIDATION:
#   Evaluates model generalization by splitting data into K folds.
#   - n_splits=5: data is split into 5 parts; model trains on 4, tests on 1, repeated 5 times
#   - shuffle=True: shuffles data before splitting (important for ordered datasets like Iris)
#   - Returns 5 accuracy scores (one per fold); mean gives overall model performance estimate

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
X, y = data.data, data.target

model = RandomForestClassifier(random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())
