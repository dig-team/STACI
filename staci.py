from confident_dtree import DTree
from staci_utils import *


class STACISurrogates:

    def __init__(self, max_depth, beta=1, weighted=False, prune=True):
        self.trees = {}
        self.max_depth = max_depth
        self.beta = beta
        self.weighted = weighted
        self.prune = prune

    def fit(self, X, y, features, target):
        """

        :param X: array-like of shape (n_samples, n_features)
            The training input samples.
        :param y: array-like of shape (n_samples,). The target values (class labels) as integers
        :param bb_model: Black box model to explain (must have predict() method)
        :param features: Attribute (column) names
        :param target: The label column name
        :return:

        """
        data = data_preparation(X, y, features, target)
        weights = compute_weights(data, target, self.weighted)
        for class_label in data[target].unique():
            self.trees[class_label] = DTree(beta=self.beta, max_depth=self.max_depth)
            self.trees[class_label].fit(data, features, class_label, target, weights)
            if self.prune:
                self.__prune()
        return self

    def __prune(self):
        tree_depth = self.max_depth

        while tree_depth > 0:
            for class_label, tree in self.trees.items():
                nodes_to_remove = []
                nodes_to_add = []
                for node in tree.nodes:
                    if isinstance(node, InternalNode) and isinstance(node.child_left, LeafNode) \
                            and isinstance(node.child_right, LeafNode):
                        if node.child_right.function == node.child_left.function:
                            tree.nodes.remove(node.child_right)
                            tree.nodes.remove(node.child_left)
                            new_leaf_node = LeafNode(level=node.depth, n_samples=node.n_samples, node_id=node.node_id)
                            for parent in tree.nodes:
                                if isinstance(parent, InternalNode):
                                    if parent.child_left == node:
                                        parent.child_left = new_leaf_node
                                    elif parent.child_right == node:
                                        parent.child_right = new_leaf_node
                            new_leaf_node.values = node.values
                            new_leaf_node.function = max(node.values.items(), key=operator.itemgetter(1))[0]
                            nodes_to_remove.append(node)
                            nodes_to_add.append(new_leaf_node)

                for n_remove in nodes_to_remove:
                    tree.nodes.remove(n_remove)
                for n_add in nodes_to_add:
                    tree.nodes.append(n_add)

            tree_depth -= 1
        return self

    def predict(self, X, bb_model):
        y_pred = []
        for i in range(X.shape[0]):
            bb_model_prediction = bb_model.predict([X.iloc[i, :]])
            tree_prediction = None
            for key, tree in self.trees.items():
                surrogate_prediction = tree.predict_single(X.iloc[i, :])
                if bb_model_prediction == surrogate_prediction:
                    tree_prediction = surrogate_prediction

            if tree_prediction is None:
                tree_prediction = self.trees[bb_model_prediction[0]].predict_single(X.iloc[i, :])

            y_pred.append(tree_prediction)

        return y_pred

    def confidence_predict(self, X, count):
        y_pred = []
        confidence_tot = []
        gen = []
        exp_len = []
        for i in range(X.shape[0]):
            max_confidence = 0.0
            best_prediction = None
            v = 0
            best_path = 0
            for key, tree in self.trees.items():
                surrogate_prediction = tree.predict_single(X.iloc[i, :])
                path = tree.decision_path(X.iloc[i, :])
                confidence = compute_confidence(tree, path, X.iloc[i, :])
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_prediction = surrogate_prediction
                    best_path = len(path) - 1
                    for node in tree.nodes:
                        if node.node_id == path[-1]:
                            v = max(node.values.values()) / count[best_prediction]
            exp_len.append(best_path)
            y_pred.append(best_prediction)
            confidence_tot.append(max_confidence)
            if v > 0:
                gen.append(v)

        return y_pred, confidence_tot, gen, exp_len

    def verbose_predict(self, x, true_prediction):

        max_confidence = 0.0
        best_tree = None
        best_path = None
        true_samples = 0
        total = 0

        for key, tree in self.trees.items():
            surrogate_prediction = tree.predict_single(x)
            if surrogate_prediction == true_prediction:
                path = tree.decision_path(x)
                confidence, n_samples = compute_confidence_leaf(tree, path, x)
                if n_samples > 0:
                    if (confidence / n_samples) > max_confidence:
                        max_confidence = confidence / n_samples
                        true_samples = confidence
                        total = n_samples
                        best_tree = tree
                        best_path = path
        explanation = []
        if best_tree and best_path:
            for node_id in best_path:
                node = best_tree.nodes[node_id]
                if isinstance(node, InternalNode):
                    if x[node.feature] <= node.threshold:
                        if len(explanation) > 0:
                            added = False
                            for j in range(len(explanation)):
                                if node.feature in explanation[j]:
                                    if "<=" in explanation[j]:
                                        if best_tree.nodes[best_path[j]].threshold > node.threshold:
                                            explanation.pop(j)
                                            explanation.append(node.feature + " <= " + str(node.threshold))
                                            added = True
                                        else:
                                            added = True
                                    elif ">" in explanation[j]:
                                        explanation.pop(j)
                                        explanation.append(
                                            str(best_tree.nodes[best_path[j]].threshold) + " < " + node.feature + " <= "
                                            + str(node.threshold))
                                        added = True
                            if not added:
                                explanation.append(node.feature + " <= " + str(node.threshold))
                                added = True
                        else:
                            explanation.append(node.feature + " <= " + str(node.threshold))
                    else:
                        if len(explanation) > 0:
                            added = False
                            for j in range(len(explanation)):
                                if node.feature in explanation[j]:
                                    if ">" in explanation[j]:
                                        if best_tree.nodes[best_path[j]].threshold <= node.threshold:
                                            explanation.pop(j)
                                            explanation.append(node.feature + " > " + str(node.threshold))
                                            added = True
                                        else:
                                            added = True
                                    elif "<=" in explanation[j]:
                                        explanation.pop(j)
                                        explanation.append(str(node.threshold) + " < " + node.feature + " <= " + str(
                                                best_tree.nodes[best_path[j]].threshold))
                                        added = True
                            if not added:
                                explanation.append(node.feature + " > " + str(node.threshold))
                        else:
                            explanation.append(node.feature + " > " + str(node.threshold))

        else:
            return None, (true_samples, total)

        if total > 0:
            return explanation, true_samples / total
        else:
            return explanation, true_samples / (total + 1)

