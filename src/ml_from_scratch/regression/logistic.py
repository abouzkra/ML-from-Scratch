from ..base import Classifier
import numpy as np

class LogisticRegression(Classifier):
    def __init__(self, alpha=0.001, max_iter=1000):
        self.W = None
        self.b = .0
        self.alpha = alpha
        self.max_iter = max_iter
        self.cost_updates = []

    def fit(self, X, Y):
        self.W = None
        self.b = .0
        self._gradient_descent(X, Y)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict_prob(self, X):
        return self.sigmoid(X @ self.W + self.b)

    def predict(self, X=None, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)

    def cost(self, X, Y):
        p = np.clip(self.predict_prob(), 1e-15, 1 - 1e-15)
        return -np.sum(
            self.Y * np.log(p) + \
            (1 - self.Y) * np.log(1-p)
            ) / len(self.Y)

    def _gradient_descent(self, X, Y):
        m = len(Y)
        for _ in range(self.max_iter):
            errors = self.predict_prob() - Y
            dW = (X.T @ errors) / m
            db = np.sum(errors) / m
            self.W -= self.alpha * dW
            self.b -= self.alpha * db
            self.cost_updates[_] = self.cost()
