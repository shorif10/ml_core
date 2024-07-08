import numpy as np
import matplotlib.pyplot as plt
from supervised.regression.linear_regression_scratch import LinearRegressionScratch
from supervised.regression.linear_regression_prebuilt import LinearRegressionPrebuilt

# Generate some example data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

# Manual implementation
model_scratch = LinearRegressionScratch(learning_rate=0.01, n_iterations=100)
model_scratch.fit(X, y)
predictions_scratch = model_scratch.predict(X)

# Library implementation
model_prebuilt = LinearRegressionPrebuilt()
model_prebuilt.fit(X, y)
predictions_prebuilt = model_prebuilt.predict(X)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, predictions_scratch, color='red', label='Predicted (Scratch)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Manual Linear Regression')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, predictions_prebuilt, color='green', label='Predicted (Prebuilt)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Library Linear Regression')
plt.legend()

plt.show()
