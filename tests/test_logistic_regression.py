import unittest
import numpy as np
from supervised.classification.logistic_regression_scratch import LogisticRegressionScratch
from supervised.classification.logistic_regression_prebuilt import LogisticRegressionPrebuilt


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1], [2], [3], [4], [5]])
        self.y = np.array([0, 0, 1, 1, 1])  # Binary classification labels

    def test_scratch_implementation(self):
        model = LogisticRegressionScratch(learning_rate=0.01, n_iterations=1000)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        np.testing.assert_array_almost_equal(predictions, self.y, decimal=1)

    def test_prebuilt_implementation(self):
        model = LogisticRegressionPrebuilt()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        np.testing.assert_array_almost_equal(predictions, self.y)


if __name__ == '__main__':
    unittest.main()
