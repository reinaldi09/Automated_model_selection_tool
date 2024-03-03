import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetLoader:
    def __init__(self, file_path, test_size=0.3, random_state=42):
        self.file_path = file_path
        self.test_size = test_size
        self.random_state = random_state

    def load_dataset(self):
        dataset = pd.read_csv(self.file_path) # Load dataset from CSV file
        X = dataset.drop(columns=['Target'])
        y = dataset['Target']

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test
