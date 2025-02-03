"""-------------------------------------------------------
KNN: Handwriting Recognition using K-Nearest Neighbors (KNN)
-------------------------------------------------------
Author:  JD
ID:      91786
Uses:    numpy,matplotlib,Scikit-Learn
Version:  1.0.8
__updated__ = Sun Feb 02 2025
-------------------------------------------------------
"""

# IMPORTS
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load MNIST Data
X, y = fetch_openml("mnist_784", return_X_y=True)
# convert labels into intergers
y = (y.astype(int)).to_numpy()
X = X.to_numpy()

# plt.imshow(X[0].reshape(28, 28), cmap="gray")
# plt.title(f"Label: {y[0]}")
# plt.show()


# NORMAILZE
X = X / 255.0

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define KNN model with K=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy:.4f}")

# Show classification report
print(classification_report(y_test, y_pred))
