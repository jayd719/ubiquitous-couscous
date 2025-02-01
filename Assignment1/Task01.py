"""-------------------------------------------------------
CP322: Assignment 1- Task One
-------------------------------------------------------
Author:  JD
ID:      91786
Uses:    numpy,openCV,matplotlib,pandas,OpenCV
Version:  1.0.8
__updated__ = Thu Jan 30 2025
-------------------------------------------------------
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
X = X[:, [2]]
print(X)
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor = LinearRegression().fit(X_train, y_train)


from sklearn.metrics import mean_squared_error, r2_score

y_pred = regressor.predict(x_test)

print(f"Root Mean Square: {mean_squared_error(y_test,y_pred):.4f}")
print(f"R2 Score: {r2_score(y_test,y_pred):.4f}")


import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 5), sharex=True, sharey=True)

ax[0, 0].scatter(X_train, y_train, label="Train data points")
ax[0, 0].plot(
    X_train,
    regressor.predict(X_train),
    linewidth=3,
    color="tab:orange",
    label="Model predictions",
)
ax[0, 0].set(xlabel="Feature", ylabel="Target", title="Train set")
ax[0, 0].legend()

ax[0, 1].scatter(x_test, y_test, label="Test data points")
ax[0, 1].plot(
    x_test, y_pred, linewidth=3, color="tab:orange", label="Model predictions"
)
ax[0, 1].set(xlabel="Feature", ylabel="Target", title="Test set")
ax[0, 1].legend()


fig.suptitle("Linear Regression")

plt.show()
