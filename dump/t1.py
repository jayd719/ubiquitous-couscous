import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate data
np.random.seed(42)  # For reproducibility
x = np.random.randint(100, size=(100, 1))
true_slope = np.random.choice(
    np.arange(-10, 11)[np.arange(-10, 11) != 0]
)  # Exclude zero
true_intercept = 10
noise = np.random.normal(0, 40, size=(100, 1))
y = true_slope * x + true_intercept + noise

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(xtrain, ytrain)
y_pred = model.predict(xtest)

# Sort test data for line plotting
xtest_sorted_idx = np.argsort(xtest.flatten())
xtest_sorted = xtest[xtest_sorted_idx]
y_pred_sorted = y_pred[xtest_sorted_idx]

# Compute metrics

# Plot results
fig, ax = plt.subplots(ncols=2, figsize=(12, 6))

# Scatter plot of training and test data
ax[0].scatter(xtrain, ytrain, label="Training Data", color="red", alpha=0.6)
ax[0].scatter(xtest, ytest, label="Testing Data", color="blue", alpha=0.6)
ax[0].plot(
    xtest_sorted, y_pred_sorted, label="Prediction Line", color="black", linewidth=2
)
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")
ax[0].set_title("Linear Regression Fit")
ax[0].legend()

# Bar plot for MSE and R² score
ax[1].bar(
    ["MSE", "R² Score"],
    [accuracy_score(ytest, y_pred), accuracy_score(ytest, y_pred)],
    color=["green", "purple"],
)
ax[1].set_title("Model Performance Metrics")

plt.tight_layout()
plt.show()
