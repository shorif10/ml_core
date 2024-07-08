import unittest
import numpy as np
from supervised.regression.linear_regression_scratch import LinearRegressionScratch
from supervised.regression.linear_regression_prebuilt import LinearRegressionPrebuilt


class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1], [2], [3], [4], [5]])
        self.y = np.array([1, 2, 3, 4, 5])

    def test_scratch_implementation(self):
        model = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        np.testing.assert_array_almost_equal(predictions, self.y, decimal=1)

    def test_prebuilt_implementation(self):
        model = LinearRegressionPrebuilt()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        np.testing.assert_array_almost_equal(predictions, self.y, decimal=1)


if __name__ == '__main__':
    unittest.main()
