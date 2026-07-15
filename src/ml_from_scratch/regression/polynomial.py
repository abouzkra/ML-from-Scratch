from ..base import Regressor
import numpy as np


class PolynomialRegression(Regressor):
    def __init__(self, degree=2, alpha=0.001, max_iter=1000, method='GD'):
        self.method = method
        self.degree = degree
        self.alpha = alpha
        self.max_iter = max_iter
        self.b = .0
        self.W = None
        self.cost_updates = np.zeros(self.max_iter)

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y).flatten()
        if X.ndim == 1: X = X.reshape(-1, 1)

        X_expanded = self._expand_poly_features(np.asarray(X))
        self.b = .0
        self.W = np.zeros(X_expanded.shape[1])

        if self.method == 'GD':
            self._gradient_descent(X_expanded, Y)
        elif self.method == 'OLS':
            self._ols(X_expanded, Y)
        else:
            raise ValueError(f"Invalid fitting method '{self.method}': 'GD' or 'OLS")

    def _expand_poly_features(self, X):
        X_expanded = X
        for i in range(2, self.degree + 1):
            X_expanded = np.c_[X_expanded, X ** i]
        return X_expanded

    def predict(self, x):
        if x.shape[1] != self.degree:
            x = self._expand_poly_features(x)
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
        weights = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ Y
        self.b = weights[0]
        self.W = weights[1:]
