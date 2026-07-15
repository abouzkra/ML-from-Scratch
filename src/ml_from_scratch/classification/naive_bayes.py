from ..base import Classifier
import numpy as np


class NaiveBayesClassifier(Classifier):
    def __init__(self):
        self.classes = None
        self.n_classes = 0
        self.n_features = 0
        self.priors = None
        # For Multinomial or Categorical
        self.likelihoods = None
        # If Gaussian:
        self.means = None
        self.vars = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]

        self.priors = np.array([np.sum(y == cls)/y.shape[0] for cls in self.classes])
        self.likelihoods = np.zeros((self.n_classes, self.n_features))
        self.means = np.zeros((self.n_classes, self.n_features))
        self.vars = np.zeros((self.n_classes, self.n_features))

        for i, cls in enumerate(self.classes):
            X_c = X[y == cls]
            self.means[i, :] = np.mean(X_c, axis=0)
            self.vars[i, :] = np.var(X_c, axis=0)

    def _probability_density(self, class_idx, x):
        eps = 1e-6
        mean = self.means[class_idx]
        var = self.vars[class_idx] + eps
        numerator = np.exp(-np.power((x - mean), 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator

    def predict(self, X):
        eps = 1e-6
        posteriors = np.zeros((X.shape[0], self.n_classes))

        for i, cls in enumerate(self.classes):
            prior = np.log(self.priors[i])
            likelihood = np.sum(np.log(self._probability_density(i, X)) + eps, axis=1)
            posteriors[:, i] = prior + likelihood

        return self.classes[np.argmax(posteriors, axis=1)]
