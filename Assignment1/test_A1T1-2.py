from functions import TEST_DATA, preProcess, plot_results
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

from sklearn.datasets import (
    load_iris,
    load_digits,
    load_wine,
    load_breast_cancer,
    load_diabetes,
    fetch_california_housing,
)

# TEST 1
print("Testing IRIS DATASET")
X, y = load_iris(return_X_y=True)
preProcess(X)
k_values, avg_accuracy = TEST_DATA(X, y)
plot_results(k_values, avg_accuracy, "IRIS")

# TEST 2
print("Testing DIABETES DATASET")
X, y = load_diabetes(return_X_y=True)
preProcess(X)
k_values, avg_accuracy = TEST_DATA(X, y)
plot_results(k_values, avg_accuracy, "DIABETES")

# TEST 3
print("Testing DIGITS DATASET")
X, y = load_digits(return_X_y=True)
preProcess(X)
k_values, avg_accuracy = TEST_DATA(X, y)
plot_results(k_values, avg_accuracy, "DIGITS")

# TEST 4
print("Testing WINE DATASET")
X, y = load_wine(return_X_y=True)
preProcess(X)
k_values, avg_accuracy = TEST_DATA(X, y)
plot_results(k_values, avg_accuracy, "WINE")

# TEST 5
print("Testing BREAST CANCER DATASET")
X, y = load_breast_cancer(return_X_y=True)
preProcess(X)
k_values, avg_accuracy = TEST_DATA(X, y)
plot_results(k_values, avg_accuracy, "BREAST CANCER")

# TEST 6 (Regression Task)
print("Testing CALIFORNIA HOUSING DATASET")
X, y = fetch_california_housing(return_X_y=True)
preProcess(X)
k_values, avg_accuracy = TEST_DATA(X, y)
plot_results(k_values, avg_accuracy, "CALIFORNIA HOUSING")
