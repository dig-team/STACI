from cf_tree import DTree
from cf_utils import *


class CFSurrogates:

    def __init__(self, max_depth, beta, weighted=False):
        self.trees = {}
        self.max_depth = max_depth
        self.beta = beta
        self.weighted = weighted

    def fit(self, X, y, bb_model, features, target):
        data = data_preparation(X, y, features, target)
        weights = compute_weights(data, target, self.weighted)
        for class_label in data[target].unique():
            self.trees[class_label] = DTree(beta=self.beta, max_depth=self.max_depth)
            self.trees[class_label].fit(data, bb_model, features, class_label, target, weights)
            """
            if len(self.trees[class_label].nodes) == 1 and self.beta < 0.9:
                new_beta = self.beta
            while len(self.trees[class_label].nodes) == 1 and new_beta < 0.9:
                print("Increasing beta for tree class ", class_label)
                new_beta += 0.1
                self.trees[class_label] = DTree(beta=new_beta, max_depth=self.max_depth)
                self.trees[class_label].fit(data, bb_model, features, class_label, target, weights)
            """
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

    def confidence_predict(self, X):
        y_pred = []
        confidence_tot = []

        for i in range(X.shape[0]):
            max_confidence = 0.0
            best_prediction = None
            for key, tree in self.trees.items():
                surrogate_prediction = tree.predict_single(X.iloc[i, :])
                path = tree.decision_path(X.iloc[i, :])
                confidence = compute_confidence(tree, path, X.iloc[i, :])
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_prediction = surrogate_prediction
            # print("Data, confidence: ", X.iloc[i, :], max_confidence)
            y_pred.append(best_prediction)
            confidence_tot.append(max_confidence)

        return y_pred, confidence_tot
