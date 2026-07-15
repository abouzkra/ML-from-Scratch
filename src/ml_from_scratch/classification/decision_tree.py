from ..base import Classifier, Regressor
import numpy as np


class TreeNode:
    def __init__(self, feature=None, value=None, children=None, gain=None, leaf=False, fallback=None):
        self.feature = feature
        self.value = value
        self.children = children
        self.gain = gain
        self.leaf = leaf

        # used if unseen discrete data appears during prediction
        self.fallback = fallback

    def __repr__(self):
        if self.leaf:
            return f"Leaf: {self.value}"
        if self.value is not None: # This is a continuous split node
            return f"Feature {self.feature} (threshold: {self.value})"
        return f"Feature {self.feature} (discrete)"


class DecisionTree(Classifier):
    def __init__(self, min_split_samples=2, max_depth=100):
        self.classes = None
        self.tree_root = None
        self.min_split_samples = min_split_samples
        self.max_depth = max_depth

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.tree_root = self._build_tree(X, list(range(X.shape[1])), y, 0, None)

    def _is_discrete(self, feature_col):
        if np.issubdtype(feature_col.dtype, np.number):
            return (np.unique(feature_col).shape[0] / feature_col.shape[0]) <= 0.05
        return True

    def _label_prob(self, y, label):
        return y[y == label].shape[0] / y.shape[0]

    def _majority_decision(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / y.shape[0]
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _information_gain(self, y, y_subsets):
        entropy_before = self._entropy(y)
        n = y.shape[0]
        entropy_after = np.sum([
            (len(sub) / n) * self._entropy(sub)
            for sub in y_subsets if sub.shape[0] > 0
            ])

        return entropy_before - entropy_after

    def _best_split_feature(self, X, features, y):
        best_feature = None
        threshold = None
        best_gain = -1

        for feature in features:
            col = X[:, feature]

            if self._is_discrete(col):
                y_subsets = [y[col == v] for v in np.unique(col)]
                gain = self._information_gain(y, y_subsets)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    threshold = None

            else:
                sorted_indices = np.argsort(col)
                sorted_vals = col[sorted_indices]
                sorted_y = y[sorted_indices]
                unique_vals = np.unique(sorted_vals)

                for i in range(1, unique_vals.shape[0]):
                    v1, v2 = unique_vals[i - 1], unique_vals[i]

                    midpoint = (v1 + v2) / 2.0
                    y_right = y[col >= midpoint]
                    y_left = y[col < midpoint]

                    gain = self._information_gain(y, [y_left, y_right])

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        threshold = midpoint

        return best_feature, threshold, best_gain


    def _build_tree(self, X, features, y, depth, parent_majority):
        if X.shape[0] == 0:
            return TreeNode(value=parent_majority, leaf=True)

        majority = self._majority_decision(y)
        best_feature, threshold, best_gain = self._best_split_feature(X, features, y)

        if (np.unique(y).shape[0] == 1 or len(features) == 0 or depth >= self.max_depth
            or X.shape[0] < self.min_split_samples or best_gain <= 0):
            return TreeNode(value=majority, leaf=True)

        col = X[:, best_feature]
        if threshold is None:
            unique_vals = np.unique(col)
            next_features = [f for f in features if f != best_feature]

            return TreeNode(
                feature=best_feature,
                children={
                v: self._build_tree(X[col == v], next_features, y[col == v], depth + 1, majority)
                for v in unique_vals
                },
                gain=best_gain, fallback=majority
            )
        else:
            return TreeNode(
            feature=best_feature, value=threshold,
            children={
                True: self._build_tree(X[col >= threshold], features, y[col >= threshold], depth + 1, majority),
                False:  self._build_tree(X[col < threshold], features, y[col < threshold], depth + 1, majority)
            },
            gain=best_gain, fallback=majority
            )

    def _predict_one(self, x, node: TreeNode):
        if node.leaf:
            return node.value

        if node.value is None: # Discrete node
            feature_val = x[node.feature]
            if feature_val in node.children:
                return self._predict_one(x, node.children[feature_val])
            else:
                return node.fallback

        else: # Continuous node
            return self._predict_one(x, node.children[x[node.feature] >= node.value])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_root) for x in X])

    # This was vibecoded don't bother if it doesn't work correctly hahaha
    def pretty_print_tree(self, node=None, prefix="", is_last=True, edge_label=""):
        if self.tree_root is None:
            print("No tree was fitted!")
            return

        if node is None:
            node = self.tree_root

        marker = "└── " if is_last else "├── "
        label_str = f"({edge_label}) " if edge_label != "" else ""

        print(prefix + marker + label_str + str(node))
        new_prefix = prefix + ("    " if is_last else "│   ")

        if node.children:
            child_items = list(node.children.items())
            for i, (value, child_node) in enumerate(child_items):
                last_child = (i == len(child_items) - 1)

                friendly_label = str(value)
                if node.value is not None:
                    friendly_label = f">= {node.value}" if value else f"< {node.value}"

                self.pretty_print_tree(child_node, new_prefix, last_child, friendly_label)
