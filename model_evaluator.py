import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r_squared = self.model.score(X_test, y_test)
        # Additional evaluation metrics can be calculated here
        return mse, r_squared

    def visualize_results(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs. Predicted Values')
        plt.show()