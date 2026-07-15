from ..base import Classifier, Regressor
from abc import ABC, abstractmethod
import numpy as np


class KNNBase(ABC):
    def __init__(self, k=5, p=2):
        self.k = k
        self.p = p

    def fit(self, X, y):
        self.X = X
        self.y = y

    @abstractmethod
    def _get_single_prediction(self, x):
        pass

    def _dist(self, x1, x2):
        if self.p == np.inf:
            return np.max(np.abs(x1 - x2), axis=1)

        return np.sum(np.abs(x1 - x2) ** self.p, axis=1) ** (1 / self.p)

    def predict(self, X_test):
        return np.apply_along_axis(self._get_single_prediction, 1, X_test)


class KNNClassifier(KNNBase, Classifier):
    def _get_single_prediction(self, x):
        knn_idx = np.argsort(self._dist(x, self.X))[:self.k] # indexes of the k nearest neighbors
        knn_labels = self.y[knn_idx] # classifications of the k nearest neighbors
        classes, class_occs = np.unique(knn_labels, return_counts=True)
        return classes[np.argmax(class_occs)]


class KNNRegressor(KNNBase, Regressor):
    def _get_single_prediction(self, x):
        knn_idx = np.argsort(self._dist(x, self.X))[:self.k] # indexes of the k nearest neighbors
        knn_labels = self.y[knn_idx] # classifications of the k nearest neighbors
        return np.mean(knn_labels)
