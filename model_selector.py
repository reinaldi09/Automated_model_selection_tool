from sklearn.metrics import mean_squared_error

class ModelSelector:
    def __init__(self, models):
        self.models = models

    def select_best_model(self, X_train, y_train, X_test, y_test):
        best_model = None
        best_score = float('inf')  # Initialize with a high value for minimization
        for model in self.models:
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            if mse < best_score:
                best_model = model
                best_score = mse
        return best_model