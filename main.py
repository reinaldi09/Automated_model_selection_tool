from load_dataset import DatasetLoader
from model_selector import ModelSelector
from regression_model import LinearRegressionModel
from regression_model import PolynomialRegressionModel
from model_evaluator import ModelEvaluator


def main():
    # Example usage
    file_path = 'D:\Pythonproj\ML_basics\Automated_model_selection_tool\dataset.csv'
    dataset_loader = DatasetLoader(file_path)
    X_train, X_test, y_train, y_test = dataset_loader.load_dataset()

    linear_regression_model = LinearRegressionModel()
    polynomial_regression_model = PolynomialRegressionModel(degree=2)

    model_selector = ModelSelector([linear_regression_model, polynomial_regression_model])
    best_model = model_selector.select_best_model(X_train, y_train, X_test, y_test)
    print(best_model)

    model_evaluator = ModelEvaluator(best_model)
    # mse, r_squared = model_evaluator.evaluate(X_test, y_test)
    # print(f'MSE: {mse}, R-squared: {r_squared}')

    model_evaluator.visualize_results(X_test, y_test)

if __name__ == "__main__":
    main()