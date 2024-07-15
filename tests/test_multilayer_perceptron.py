import unittest
import numpy as np
from supervised.classification.multilayer_perceptron_scratch import NeuralNetworkMultilayerScratch
from supervised.classification.multilayer_perceptron_prebuilt import NeuralNetworkMultilayerPrebuilt


class TestNeuralNetworkMultilayer(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])  # XOR problem

    def test_scratch_implementation(self):
        model = NeuralNetworkMultilayerScratch(input_size=2, hidden_sizes=[3, 2], output_size=1, learning_rate=0.1)
        model.fit(self.X, self.y, epochs=10000)
        predictions = model.predict(self.X)
        np.testing.assert_array_almost_equal(predictions, self.y, decimal=1)

    def test_prebuilt_implementation(self):
        model = NeuralNetworkMultilayerPrebuilt(input_size=2, hidden_sizes=[3, 2], output_size=1)
        model.fit(self.X, self.y, epochs=10000)
        predictions = model.predict(self.X)
        np.testing.assert_array_almost_equal(predictions, self.y, decimal=1)


if __name__ == '__main__':
    unittest.main()
