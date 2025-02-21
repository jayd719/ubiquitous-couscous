#!/usr/bin/env python
# coding: utf-8

"""
Assignment 2: Linear Regression

Linear regression is a fundamental supervised learning algorithm used in machine learning and statistics.
It models the relationship between a dependent variable (target) and one or more independent variables (features)
using a linear equation.
"""

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

# Define data directory path
dir_path = "./Assignment2/regression-dataset/"
assert os.path.exists(
    dir_path
), "Error: Data directory not found. Please check dir_path."

# Load training and testing datasets
X_train = pd.read_csv(os.path.join(dir_path, "train_inputs.csv"), header=None)
y_train = pd.read_csv(os.path.join(dir_path, "train_targets.csv"), header=None)
X_test = pd.read_csv(os.path.join(dir_path, "test_inputs.csv"), header=None)
y_test = pd.read_csv(os.path.join(dir_path, "test_targets.csv"), header=None)

# Display dataset shapes
print("Dataset Shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}\n")

# Display first few rows of the datasets
print("Sample Data (X_train):")
print(X_train.head(), "\n")
print("Sample Data (y_train):")
print(y_train.head(), "\n")

# Convert data to numpy arrays for compatibility with Scikit-Learn
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.to_numpy())

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)  # Adjust the number of components as needed
X_train = pca.fit_transform(X_train)
y_train = y_train.to_numpy().flatten()
X_test = scaler.transform(X_test.to_numpy())
X_test = pca.transform(X_test)
y_test = y_test.to_numpy().flatten()

# Confirm data transformation
print("Transformed Data Shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}\n")

# Exploratory Data Analysis (EDA)
plt.style.use("ggplot")  # Use a visually appealing style
fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
fig.suptitle("Data Visualization for Linear Regression", fontsize=16, fontweight="bold")

ax[0].scatter(X_train[:, 0], y_train, alpha=0.5, color="blue")
ax[0].set_xlabel("Feature 1")
ax[0].set_ylabel("Target")
ax[0].set_title("Feature 1 vs Target")

ax[1].scatter(X_train[:, 1], y_train, alpha=0.5, color="red")
ax[1].set_xlabel("Feature 2")
ax[1].set_title("Feature 2 vs Target")

plt.tight_layout()
plt.show()

# Hyperparameter Settings
KFOLD = 10  # Number of cross-validation folds
LEARNING_RATE_RANGE = (0, 1)  # Range of learning rate values to test
INCREMENT_VALUE = 0.1  # Step size for learning rate iteration

# Hyperparameter tuning using cross-validation
alpha = LEARNING_RATE_RANGE[0]
results = {}

for alpha in np.arange(
    LEARNING_RATE_RANGE[0], LEARNING_RATE_RANGE[1] + INCREMENT_VALUE, INCREMENT_VALUE
):
    model = SGDRegressor(
        alpha=0.001, learning_rate="constant", eta0=alpha, max_iter=1000, tol=1e-3
    )
    scores = -cross_val_score(
        model, X_train, y_train, cv=KFOLD, scoring="neg_mean_squared_error"
    )
    results[alpha] = scores.mean()
    alpha += INCREMENT_VALUE

# Display results
print("Hyperparameter Tuning Results (MSE):")
for alpha, mse in results.items():
    print(f"Alpha: {alpha:.2f}, MSE: {mse:.4f}")
