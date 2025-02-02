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

plt.style.use("ggplot")
plt.figure(figsize=(12, 8))

# TEST 1
X, y = load_iris(return_X_y=True)
preProcess(X)
k_values, avg_accuracy = TEST_DATA(X, y)
plot_results(k_values, avg_accuracy, "IRIS")
plt.plot(
    k_values,
    avg_accuracy,
    marker="o",
    linestyle="dashed",
    label="IRIS",
    alpha=0.7,
    linewidth=1.5,
)


# TEST 2
X, y = load_diabetes(return_X_y=True)
preProcess(X)
k_values, avg_accuracy = TEST_DATA(X, y)
plot_results(k_values, avg_accuracy, "DIABETES")
plt.plot(
    k_values,
    avg_accuracy,
    marker="o",
    linestyle="dashed",
    label="diabetes",
    alpha=0.7,
    linewidth=1.5,
)


# savd

outputfile = "all"
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validation Accuracy")
plt.title(
    f"KNN Cross-Validation Accuracy vs k on {outputfile} Dataset",
    fontsize=16,
    fontweight="bold",
)
plt.legend(prop={"size": 12, "weight": "bold"}, labelcolor="black")
plt.axis("equal")
plt.savefig(f"{outputfile}.png", format="png", bbox_inches="tight", transparent=True)
plt.close()
