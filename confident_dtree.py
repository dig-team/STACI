from cf_nodes import *
from staci_utils import grow_with_stop
import operator


class DTree:
    """
    Confident Decision Tree class.

    """

    def __init__(self, max_depth, beta, root=None):
        self.root = root
        self.max_depth = max_depth
        self.min_samples_split = 2
        self.beta = beta
        self.nodes = []

    def fit(self, data, bb_model, features, important_class, label, weights):
        level = 0
        if data.shape[0] >= 2 and data[label].nunique() > 1:
            if self.max_depth is not None:
                if self.max_depth > 0:
                    maximum_f1 = 0.0
                    self.root = grow_with_stop(data, features, bb_model, important_class, self.max_depth, label,
                                               self.beta, self.nodes, level, weights, maximum_f1)
                else:
                    self.root = LeafNode(data.shape[0], level=level, node_id=0)
                    for l in sorted(data[label].unique()):
                        self.root.values[l] = data[data[label] == l].shape[0]
                    self.root.function = max(self.root.values.items(), key=operator.itemgetter(1))[0]
                    self.nodes.append(self.root)

            else:
                maximum_f1 = 0.0
                self.root = grow_with_stop(data, features, bb_model, important_class, self.max_depth, label, self.beta,
                                           self.nodes, level, weights, maximum_f1)
        else:
            self.root = LeafNode(data.shape[0], level=level, node_id=0)
            for l in sorted(data[label].unique()):
                self.root.values[l] = data[data[label] == l].shape[0]
            self.root.function = max(self.root.values.items(), key=operator.itemgetter(1))[0]
            self.nodes.append(self.root)

        return self

    def predict(self, X):
        y = []
        for i in range(X.shape[0]):
            y.append(self.root.predict(X.iloc[i, :]))

        return y

    def predict_single(self, x):

        return self.root.predict(x)

    def decision_path(self, x):

        path = []
        self.root.predict_verbose(x, path)

        return path
