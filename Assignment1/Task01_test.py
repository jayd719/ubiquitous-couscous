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

os.system("clear" if os.name == "posix" else "cls")


class KNN_Classifier:
    def __init__(self, k=2, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric.lower()
        # list of available metircs
        self.distance_metrics = ["euclidean", "manhattan"]

        # check if provided distance metric in list
        assert (
            self.distance_metric in self.distance_metrics
        ), f"Invalid Distance Metric\nAviable Metrics:\n\t{self.get_metric_list()}"

    def get_metric_list(self):
        return "\n\t".join(value for value in self.distance_metrics)

    def __str__(self):
        return f"(K={self.k},Distance Metric={self.distance_metric.upper()})"

    def get_distance(self, train_data, test_data):
        distance_functions = {
            "euclidean": lambda: cal_euclidean_distance(train_data, test_data),
            "manhattan": lambda: calculate_manhattan_distance(train_data, test_data),
        }
        distance = distance_functions.get(self.distance_metric)

    def nearest_neighbours(self, train_data, test_data):
        distances = []
        for t_data in train_data:
            distance = self.get_distance(t_data, test_data)
            distances.append(t_data, distance)
        distances.sort(key=lambda x: x[1])

        neighbours = []
        for j in range(self.k):
            neighbours.append(distances[j][0])
        return neighbours

    def predict(self, train_data, test_data):
        pass


if __name__ == "__main__":
    knn = KNN_Classifier(k=10)
    print(knn)
