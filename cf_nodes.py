
class InternalNode:

    def __init__(self, level, feature='', n_samples=0, node_id=None, child_left=None, child_right=None,
                 threshold=None,):

        self.node_id = node_id
        self.child_left = child_left
        self.child_right = child_right
        self.feature = feature
        self.threshold = threshold
        self.n_samples = n_samples
        self.depth = level
        self.values = {}

    def predict(self, x):

        if x[self.feature] <= self.threshold:
            return self.child_left.predict(x)
        else:
            return self.child_right.predict(x)

    def predict_verbose(self, x, path):
        path.append(self.node_id)
        if x[self.feature] <= self.threshold:
            return self.child_left.predict_verbose(x, path)
        else:
            return self.child_right.predict_verbose(x, path)


class LeafNode:

    def __init__(self, n_samples, level, node_id=None):

        self.n_samples = n_samples
        self.depth = level
        self.node_id = node_id
        self.values = {}
        self.function = None

    def predict(self, X):

        return self.function

    def predict_verbose(self, x, path):
        path.append(self.node_id)

        return path
