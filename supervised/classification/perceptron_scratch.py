import numpy as np

class NeuralNetworkScratch:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        return z * (1 - z)

    def fit(self, X, y, epochs=1000):
        for _ in range(epochs):
            # Forward pass
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self._sigmoid(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self._sigmoid(z2)

            # Backward pass
            error_output = y - a2
            d_output = error_output * self._sigmoid_derivative(a2)

            error_hidden = d_output.dot(self.W2.T)
            d_hidden = error_hidden * self._sigmoid_derivative(a1)

            # Update weights and biases
            self.W2 += a1.T.dot(d_output) * self.learning_rate
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
            self.W1 += X.T.dot(d_hidden) * self.learning_rate
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self._sigmoid(z2)
        return np.round(a2)
