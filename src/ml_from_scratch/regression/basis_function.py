from ..base import Regressor
import numpy as np


class BasisFunctionRegression(Regressor):
    def __init__(self, b_fts, alpha=0.001, max_iter=1000):
        self.b_fts = b_fts
        self.W = None
        self.b = .0
        self.alpha = alpha
        self.max_iter = max_iter
        self.cost_updates = []
    
    def fit(self, X, Y):
        self.W = np.zeros(len(self.b_fts))
        self.b = .0
        self._gradient_descent(X, Y)
    
    def _Phi(self, X):
        return np.column_stack([f(X).flatten() for f in self.b_fts])
    
    def predict(self, X=None):
        X = self.X if X is None else (X.reshape(-1, 1) if X.ndim == 1 else X)
        return self.b + self._Phi(X) @ self.W
    
    def cost(self, X, Y):
        return np.sum((self.predict(X) - Y) ** 2)/(2 * len(Y))
    
    def _gradient_descent(self, X, Y):
        m = len(Y)
        phi_matrix = self._Phi(X)
    
        for i in range(self.max_iter):
            errors = phi_matrix @ self.W + self.b - Y
            dW = (phi_matrix.T @ errors)/m
            db = np.sum(errors) / m
            self.W -= self.alpha * dW
            self.b -= self.alpha * db
            self.cost_updates = self.cost(X, Y)
