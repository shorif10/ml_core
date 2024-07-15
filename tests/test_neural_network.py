import unittest
import numpy as np
from supervised.classification.perceptron_scratch import NeuralNetworkScratch
from supervised.classification.perceptron_prebuilt import NeuralNetworkPrebuilt


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])  # XOR problem

    def test_scratch_implementation(self):
        model = NeuralNetworkScratch(input_size=2, hidden_size=3, output_size=1, learning_rate=0.1)
        model.fit(self.X, self.y, epochs=2)
        predictions = model.predict(self.X)
        np.testing.assert_array_almost_equal(predictions, self.y, decimal=1)

    def test_prebuilt_implementation(self):
        model = NeuralNetworkPrebuilt(input_size=2, hidden_size=3, output_size=1)
        model.fit(self.X, self.y, epochs=2)
        predictions = model.predict(self.X)
        np.testing.assert_array_almost_equal(predictions, self.y, decimal=1)


if __name__ == '__main__':
    unittest.main()
