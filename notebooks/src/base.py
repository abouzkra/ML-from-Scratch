from abc import ABC, abstractmethod
import numpy as np


class MLBaseModel(ABC):
    @abstractmethod
    def fit(self, X, *args, **kwargs):
        pass


class Supervised(MLBaseModel):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X, y):
        pass

    def score(self, X_test, y_test):
        return np.mean(self.predict(X_test) == y_test)


class Regressor(Supervised):
    def score(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)


class Classifier(Supervised):
    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class Unsupervised(MLBaseModel):
    @abstractmethod
    def transform(self, X):
        pass
