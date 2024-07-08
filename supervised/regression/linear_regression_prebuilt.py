from sklearn.linear_model import LinearRegression


class LinearRegressionPrebuilt:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)
