import numpy as np

class NeuralNetworkMultilayerScratch:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_sizes[0])
        self.b1 = np.zeros((1, self.hidden_sizes[0]))
        self.W2 = np.random.randn(self.hidden_sizes[0], self.hidden_sizes[1])
        self.b2 = np.zeros((1, self.hidden_sizes[1]))
        self.W3 = np.random.randn(self.hidden_sizes[1], self.output_size)
        self.b3 = np.zeros((1, self.output_size))

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
            z3 = np.dot(a2, self.W3) + self.b3
            a3 = self._sigmoid(z3)

            # Backward pass
            error_output = y - a3
            d_output = error_output * self._sigmoid_derivative(a3)

            error_hidden2 = d_output.dot(self.W3.T)
            d_hidden2 = error_hidden2 * self._sigmoid_derivative(a2)

            error_hidden1 = d_hidden2.dot(self.W2.T)
            d_hidden1 = error_hidden1 * self._sigmoid_derivative(a1)

            # Update weights and biases
            self.W3 += a2.T.dot(d_output) * self.learning_rate
            self.b3 += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
            self.W2 += a1.T.dot(d_hidden2) * self.learning_rate
            self.b2 += np.sum(d_hidden2, axis=0, keepdims=True) * self.learning_rate
            self.W1 += X.T.dot(d_hidden1) * self.learning_rate
            self.b1 += np.sum(d_hidden1, axis=0, keepdims=True) * self.learning_rate

    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self._sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = self._sigmoid(z3)
        return np.round(a3)
