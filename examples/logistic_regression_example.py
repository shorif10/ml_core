import numpy as np
import matplotlib.pyplot as plt
from supervised.classification.logistic_regression_scratch import LogisticRegressionScratch
from supervised.classification.logistic_regression_prebuilt import LogisticRegressionPrebuilt

# Generate some example data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])  # Binary classification labels

# Manual implementation
model_scratch = LogisticRegressionScratch(learning_rate=0.01, n_iterations=1000)
model_scratch.fit(X, y)
predictions_scratch = model_scratch.predict(X)

# Library implementation
model_prebuilt = LogisticRegressionPrebuilt()
model_prebuilt.fit(X, y)
predictions_prebuilt = model_prebuilt.predict(X)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Actual')
plt.scatter(X, predictions_scratch, color='red', marker='x', label='Predicted (Scratch)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Manual Logistic Regression')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Actual')
plt.scatter(X, predictions_prebuilt, color='green', marker='x', label='Predicted (Prebuilt)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Library Logistic Regression')
plt.legend()

plt.show()
