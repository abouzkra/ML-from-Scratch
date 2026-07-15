from ..base import Classifier
import numpy as np

class LogisticRegression(Classifier):
    def __init__(self, alpha=0.001, max_iter=1000):
        self.W = None
        self.b = .0
        self.alpha = alpha
        self.max_iter = max_iter
        self.cost_updates = np.zeros(max_iter)

    def fit(self, X, Y):
        self.W = np.zeros(X.shape[1])
        self.b = .0
        self._gradient_descent(X, Y)

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def predict_prob(self, X):
        return self._sigmoid(X @ self.W + self.b)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)
    
    def cost(self, X, Y):
        p = np.clip(self.predict_prob(X), 1e-15, 1 - 1e-15)
        return -np.sum(
            Y * np.log(p) + \
            (1 - Y) * np.log(1 - p)
            ) / len(Y)
    
    def _gradient_descent(self, X, Y):
        m = len(Y)
        for _ in range(self.max_iter):
          errors = self.predict_prob(X) - Y
          dW = (X.T @ errors)/m
          db = np.sum(errors)/m
          self.W -= self.alpha * dW
          self.b -= self.alpha * db
          self.cost_updates[_] = self.cost(X, Y)


# TODO: Implement multinomial regression
class MultinomialRegression(Classifier):
    def __init__(self, alpha=0.001, max_iter=1000):
        self.W = None
        self.b = .0
        self.alpha = alpha
        self.max_iter = max_iter
        self.cost_updates = np.zeros(max_iter)

    def fit(self, X, Y):
        self.W = np.zeros(X.shape[1])
        self.b = .0
        self._gradient_descent(X, Y)

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def predict_prob(self, X):
        return self._sigmoid(X @ self.W + self.b)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)
    
    def cost(self, X, Y):
        p = np.clip(self.predict_prob(X), 1e-15, 1 - 1e-15)
        return -np.sum(
            Y * np.log(p) + \
            (1 - Y) * np.log(1 - p)
            ) / len(Y)
    
    def _gradient_descent(self, X, Y):
        m = len(Y)
        for _ in range(self.max_iter):
          errors = self.predict_prob(X) - Y
          dW = (X.T @ errors)/m
          db = np.sum(errors)/m
          self.W -= self.alpha * dW
          self.b -= self.alpha * db
          self.cost_updates[_] = self.cost(X, Y)
