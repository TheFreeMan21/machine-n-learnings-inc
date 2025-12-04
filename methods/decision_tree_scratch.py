import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from util.filter_data import filtering

class DecisionTreeScratch:

    def __init__(self, max_depth=5, min_samples_split=10, min_samples_leaf=1, max_leaf_nodes=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.tree = None
        self.n_leaves = 0

    def fit(self, X, y):
        self.n_leaves = 0
        self.tree = self.build_tree(X, y, depth=0)

    def build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            self.n_leaves += 1
            return {"value": np.mean(y)}

        best_feature, best_thresh, best_score = None, None, float("inf")
        best_left, best_right = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                mse = (len(y_left)*np.var(y_left) + len(y_right)*np.var(y_right)) / len(y)

                if mse < best_score:
                    best_feature, best_thresh = feature, threshold
                    best_score = mse
                    best_left, best_right = (X[left_mask], y_left), (X[right_mask], y_right)

        if best_feature is None:
            self.n_leaves += 1
            return {"value": np.mean(y)}

        if self.max_leaf_nodes is not None and self.n_leaves >= self.max_leaf_nodes:
            self.n_leaves += 1
            return {"value": np.mean(y)}

        left_branch = self.build_tree(best_left[0], best_left[1], depth + 1)
        right_branch = self.build_tree(best_right[0], best_right[1], depth + 1)

        return {"feature": best_feature,
                "threshold": best_thresh,
                "left": left_branch,
                "right": right_branch}

    def single_pred(self, x, node):
        if "value" in node:
            return node["value"]

        if x[node["feature"]] <= node["threshold"]:
            return self.single_pred(x, node["left"])
        else:
            return self.single_pred(x, node["right"])

    def predict(self, X):
        return np.array([self.single_pred(x, self.tree) for x in X])


    def prune(self, X_val, y_val):
        self.prune_node(self.tree, X_val, y_val)

    def prune_node(self, node, X, y):
        if "value" in node:
            return

        feature = node["feature"]
        threshold = node["threshold"]

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        self.prune_node(node["left"], X[left_mask], y[left_mask])
        self.prune_node(node["right"], X[right_mask], y[right_mask])

        y_pred_keep = np.array([self.single_pred(x, node) for x in X])
        keep_error = np.mean((y - y_pred_keep)**2)

        leaf_value = np.mean(y)
        prune_error = np.mean((y - leaf_value)**2)

        if prune_error <= keep_error:
            node.clear()
            node["value"] = leaf_value
