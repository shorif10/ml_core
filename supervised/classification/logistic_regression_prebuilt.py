from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression


class LogisticRegressionPrebuilt:
    def __init__(self):
        self.model = SklearnLogisticRegression()

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)
