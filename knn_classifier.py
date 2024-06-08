import numpy as np

class KNNClassifier:
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        distances = np.linalg.norm(self.X_train - X, axis=1)
        nearest_index = np.argmin(distances)
        return self.y_train[nearest_index]
