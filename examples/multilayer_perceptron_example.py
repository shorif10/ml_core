import numpy as np
import matplotlib.pyplot as plt
from supervised.classification.multilayer_perceptron_scratch import NeuralNetworkMultilayerScratch
from supervised.classification.multilayer_perceptron_prebuilt import NeuralNetworkMultilayerPrebuilt

# Generate some example data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR problem

# Manual implementation
model_scratch = NeuralNetworkMultilayerScratch(input_size=2, hidden_sizes=[3, 2], output_size=1, learning_rate=0.1)
model_scratch.fit(X, y, epochs=5)
predictions_scratch = model_scratch.predict(X)

# Library implementation
model_prebuilt = NeuralNetworkMultilayerPrebuilt(input_size=2, hidden_sizes=[3, 2], output_size=1)
model_prebuilt.fit(X, y, epochs=5)
predictions_prebuilt = model_prebuilt.predict(X)

# Print the results
print("Scratch Implementation Predictions:", predictions_scratch)
print("Prebuilt Implementation Predictions:", predictions_prebuilt)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], marker='x', c=predictions_scratch.ravel(), cmap='coolwarm')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Manual Neural Network')
plt.legend(['Actual', 'Predicted'])

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], marker='x', c=predictions_prebuilt.ravel(), cmap='coolwarm')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Library Neural Network')
plt.legend(['Actual', 'Predicted'])

plt.show()
