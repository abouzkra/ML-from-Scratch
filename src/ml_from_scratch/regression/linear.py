from ..base import Regressor
import numpy as np


class SimpleLinearRegression(Regressor):
    def __init__(self, alpha=0.001, max_iter=1000, method='GD'):
        self.method = method
        self.max_iter = max_iter
        self.alpha = alpha
        self.cost_updates = np.zeros(self.max_iter)

    def fit(self, X, Y):
        self.w0 = 0.0
        self.w1 = 0.0
        if self.method == 'GD':
            self._gradient_descent(np.asarray(X), np.asarray(Y))
        elif self.method == 'OLS':
            self._ols(np.asarray(X), np.asarray(Y))
        else:
            raise ValueError(f"Invalid fitting method '{self.method}': 'GD' or 'OLS")

    def cost(self, X, Y):
        return np.sum((self.predict(X) - Y) ** 2) / (2 * X.shape[0])

    def predict(self, x):
        return self.w0 + self.w1 * np.asarray(x)

    def _gradient_descent(self, X, Y):
        for _ in range(self.max_iter):
            d0 = np.sum(self.predict(X) - Y) / X.shape[0]
            d1 = np.sum((self.predict(X) - Y) * X) / X.shape[0]
            self.w0 = self.w0 - self.alpha * d0
            self.w1 = self.w1 - self.alpha * d1
            self.cost_updates[_] = self.cost(X, Y)

    def _ols(self, X, Y):
        X_mean = np.mean(X)
        Y_mean = np.mean(Y)
        self.w1 = np.sum((X - X_mean) @ (Y - Y_mean)) / np.sum((X - X_mean) ** 2)
        self.w0 = Y_mean - self.w1 * X_mean


class MultipleLinearRegression(Regressor):
    def __init__(self, alpha=0.001, max_iter=1000, method='GD'):
        self.method = method
        self.alpha = alpha
        self.max_iter = max_iter
        self.cost_updates = np.zeros(self.max_iter)

    def fit(self, X, Y):
        self.b = .0
        self.W = np.zeros(X.shape[1])
        if self.method == 'GD':
            self._gradient_descent(np.asarray(X), np.asarray(Y))
        elif self.method == 'OLS':
            self._ols(np.asarray(X), np.asarray(Y))
        else:
            raise ValueError(f"Invalid fitting method '{self.method}': 'GD' or 'OLS")

    def predict(self, x):
        x = np.asarray(x)
        return x @ self.W + self.b 

    def cost(self, X, Y):
        return np.sum((self.predict(X) - Y) ** 2) / (2 * len(Y))

    def _gradient_descent(self, X, Y):
        m = len(Y)
        for _ in range(self.max_iter):
            errors = self.predict(X) - Y
            dW = (X.T @ errors) / m
            db = np.sum(errors) / m
            self.W -= self.alpha * dW
            self.b -= self.alpha * db
            self.cost_updates[_] = self.cost(X, Y)

    def _ols(self, X, Y):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        weights = np.dot(
        np.linalg.inv(X_bias.T @ X_bias),
            X_bias.T @ Y
            )
        self.b = weights[0]
        self.W = weights[1:]
