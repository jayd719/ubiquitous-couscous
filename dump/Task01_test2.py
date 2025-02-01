"""-------------------------------------------------------
CP322: Assignment 1- Task One
-------------------------------------------------------
Author:  JD
ID:      91786
Uses:    numpy
Version:  1.0.8
__updated__ = Thu Jan 30 2025
-------------------------------------------------------
"""

import os
import numpy as np
import statistics
from functions import cal_euclidean_distance, calculate_manhattan_distance
from collections import Counter

os.system("clear" if os.name == "posix" else "cls")


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        distances = [cal_euclidean_distance(x, x_train) for x_train in self.X_train]

        # get the closest k
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority voye
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
