import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm


class KNN:
    def __init__(self, k=3, dist_metric="euclidean"):
        self.k = k
        self.dist_metric = dist_metric.lower()

        self.available_metircs = ["euclidean", "manhattan"]
        assert (
            self.dist_metric in self.available_metircs
        ), f"Invalid Distance Metric\nAviable Metrics:\n\t{self.get_metric_list()}"

    def get_metric_list(self):
        return "\n\t".join(value for value in self.distance_metrics)

    def __str__(self):
        return f"(K={self.k},Distance Metric={self.dist_metric.upper()})"

    def __dimension_check(self, p1, p2):
        assert len(p1) == len(p2), "Points should have the same dimensions."

    def __euclidean(self, p1, p2):
        self.__dimension_check(p1, p2)
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def __manhattan(self, p1, p2):
        self.__dimension_check(p1, p2)
        return np.sum(np.abs(p1 - p2))

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.y_train = y_train.flatten()

    def predict(self, X_test):

        predictions = [self.__predict(x) for x in X_test]
        return predictions

    def __predict(self, x):
        distance_functions = {
            "euclidean": lambda: self.__euclidean,
            "manhattan": lambda: self.__manhattan,
        }

        algo = distance_functions.get(self.dist_metric)()
        # compute the distances
        distances = [(algo(x, xTrain), i) for i, xTrain in enumerate(self.X_train)]
        # the nearest neighbours

        distances.sort(key=lambda x: x[0])

        neighbors = [self.y_train[i] for _, i in distances[: self.k]]
        # determine label with majority vote
        most_common = Counter(neighbors).most_common(1)[0][0]
        return most_common

    def score(self, ytest, ypred):
        y_true = ytest.flatten()
        total_samples = len(y_true)
        correct_predictions = np.sum(y_true == ypred)
        return (correct_predictions / total_samples) * 100


def preProcess(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


def plot_results(k_values, avg_accuracy, outputfile="plot-one-1.png"):
    highest = k_values[np.argmax(avg_accuracy)]
    best_accuracy = max(avg_accuracy)
    print(f"Best k: {highest} with accuracy: {best_accuracy:.4f}")

    plt.style.use("ggplot")
    plt.figure(figsize=(12, 8))
    plt.plot(
        k_values,
        avg_accuracy,
        marker="o",
        linestyle="dashed",
        label="Cross-Validation Accuracy",
        alpha=0.7,
        linewidth=1.5,
        color="blue",
    )
    plt.scatter(
        highest,
        best_accuracy,
        marker="x",
        label=f"Highest k: {highest} with accuracy: {best_accuracy:.4f}",
        c="red",
        s=100,
    )
    plt.axvline(highest, linestyle="dashed", alpha=0.7, c="red", linewidth=1)
    plt.axhline(best_accuracy, linestyle="dashed", alpha=0.7, c="red", linewidth=1)
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Cross-Validation Accuracy")
    plt.title(
        f"KNN Cross-Validation Accuracy vs k on {outputfile} Dataset",
        fontsize=16,
        fontweight="bold",
    )
    plt.legend(prop={"size": 12, "weight": "bold"}, labelcolor="black")
    plt.axis("equal")
    plt.savefig(
        f"{outputfile}.png", format="png", bbox_inches="tight", transparent=True
    )
    plt.close()


def TEST_DATA(X, y):
    # Perform K Fold Cross Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=65)
    k_values = range(1, 31)
    avg_accuracy = []
    for k in tqdm(k_values):
        accuracies = []
        for train_index, test_index in kf.split(X):
            Xtrain, Xtest = X[train_index], X[test_index]
            ytrain, ytest = y[train_index], y[test_index]
            knn = KNN(k=k)
            knn.fit(Xtrain, ytrain)
            ypred = knn.predict(Xtest)
            accuracies.append(knn.score(ytest, ypred))

        avg_accuracy.append(np.mean(accuracies))
    return k_values, avg_accuracy


if __name__ == "__main__":
    X_train = pd.read_csv("knn-dataset/train_inputs.csv").to_numpy()
    x_test = pd.read_csv("knn-dataset/test_inputs.csv").to_numpy()
    y_train = pd.read_csv("knn-dataset/train_labels.csv").to_numpy()
    y_test = pd.read_csv("knn-dataset/test_labels.csv").to_numpy()

    X = np.concatenate((X_train, x_test))
    y = np.concatenate((y_train, y_test))

    preProcess(X)
    TEST_DATA(X, y)
    print("DONE")
