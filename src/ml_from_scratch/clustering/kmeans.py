from ..base import Unsupervised
import numpy as np


class KMeans(Unsupervised):
    def __init__(self, k=2, max_iter=100, save_centroid_updates=False):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.save_centroid_updates = save_centroid_updates
        self.centroid_updates = None

    def fit(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]

        # KMeans++ Initialization
        self.centroids = [X[np.random.choice(n_samples)]]
        for _ in range(1, self.k):
            min_dists = np.min(np.sum((X[:, np.newaxis] - np.array(self.centroids)) ** 2, axis=2), axis=1)
            probs = min_dists / min_dists.sum()
            self.centroids.append(X[np.random.choice(n_samples, p=probs)])
        self.centroids = np.asarray(self.centroids)

        if self.save_centroid_updates:
            self.centroid_updates = np.zeros((self.max_iter, *self.centroids.shape))

        # KMeans convergence loop
        for _ in range(self.max_iter):
            old_centroids = self.centroids
            if self.save_centroid_updates:
                self.centroid_updates[_] = old_centroids
            labels = self.transform(X)
            self.centroids = np.array([
                X[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else X[np.random.choice(n_samples)]
                for j in range(self.k)
            ])

            # Break the loop if convergence met
            if np.allclose(old_centroids, self.centroids):
                self.centroid_updates = self.centroid_updates[:_ + 1]
                break

    def transform(self, X):
        return np.argmin(
            np.sum((X[:, np.newaxis] - self.centroids) ** 2, axis=2),
            axis=1
        )
