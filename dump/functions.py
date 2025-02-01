"""-------------------------------------------------------
CP322: Assignment 1- Custom Functions Library One
-------------------------------------------------------
Author:  JD
ID:      91786
Uses:    numpy,os
Version:  1.0.8
__updated__ = Thu Jan 30 2025
-------------------------------------------------------
"""

import os
import numpy as np


def cal_euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    assert len(p1) == len(p2), "Dimension mismatch between the two points"
    dist = 0
    for i in range(len(p1)):
        dist += (p1[i] - p2[i]) ** 2
    return np.sqrt(dist)


def calculate_manhattan_distance(p1, p2):
    """Calculate the Manhattan distance between two points."""
    assert len(p1) == len(p2), "Dimension mismatch between the two points"
    dist = 0
    for i in range(len(p1)):
        dist += abs(p1[i] - p2[i])
    return dist


if __name__ == "__main__":
    os.system("clear" if os.name == "posix" else "cls")
    print("Tesing Distance Functions")
    print()
    print("Test Case 1-2D")
    p1 = (1, 2)
    p2 = (5, 6)
    distance = cal_euclidean_distance(p1, p2)
    print(f"Euclidean Distance between {p1} and {p2}: {distance:.4f}")
    distance = calculate_manhattan_distance(p1, p2)
    print(f"Manhattan Distance between {p1} and {p2}: {distance:.4f}")
    print()
    print("Test Case 2-3D")
    p1 = (1, 2, 3)
    p2 = (5, 6, 7)
    distance = cal_euclidean_distance(p1, p2)
    print(f"Euclidean Distance between {p1} and {p2}: {distance:.4f}")
    distance = calculate_manhattan_distance(p1, p2)
    print(f"Manhattan Distance between {p1} and {p2}: {distance:.4f}")
    print()
    print("Test Case 3-4D")
    p1 = (1, 2, 3, 4)
    p2 = (5, 6, 7, 8)
    distance = cal_euclidean_distance(p1, p2)
    print(f"Euclidean Distance between {p1} and {p2}: {distance:.4f}")
    distance = calculate_manhattan_distance(p1, p2)
    print(f"Manhattan Distance between {p1} and {p2}: {distance:.4f}")
