from abc import ABC, abstractmethod
import numpy as np


class MLBaseModel(ABC):
    @abstractmethod
    def fit(self, X, *args, **kwargs):
        pass


class Supervised(MLBaseModel):
    @abstractmethod
    def predict(self, *args, **kwargs):
        pass


class Regressor(Supervised):
    def score(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)


class Classifier(Supervised):
    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class Unsupervised(MLBaseModel):
    @abstractmethod
    def transform(self, *args, **kwargs):
        pass
