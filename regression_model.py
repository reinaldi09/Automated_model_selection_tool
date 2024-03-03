from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class RegressionModel:
    def __init__(self):
        pass

    def train(self, X_train, y_train):
        raise NotImplementedError("train() method must be implemented in subclass")

    def predict(self, X):
        raise NotImplementedError("predict() method must be implemented in subclass")

class LinearRegressionModel(RegressionModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

class PolynomialRegressionModel(RegressionModel):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
        self.model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)