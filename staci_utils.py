import pandas as pd
import numpy as np
from cf_nodes import *
import operator


def data_preparation(X, y, features, target):

    if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        numpy_data = np.concatenate((X, y), axis=1)
        try:
            n_rows, n_columns = y.shape
        except ValueError:
            print("Expected 2d array")
        columns = features.append(target)
        data = pd.DataFrame(numpy_data, columns=columns)
    elif isinstance(X, pd.DataFrame) and (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)):
        data = pd.concat([X, y], axis=1)
    else:
        raise TypeError("The input vectors X and y must be of the same type. Either np.ndarays or pd.Dataframes/Series")
    return data


def compute_weights(data, target, weighted):
    weights = {}
    if not weighted:
        for label in sorted(data[target].unique()):
            weights[label] = 1
    else:
        total = data.shape[0]
        for label in sorted(data[target].unique()):
            num_samples = data[data[target] == label].shape[0]
            new_num_samples = 4 * (total - num_samples)
            weights[label] = new_num_samples / num_samples

    return weights


def create_leaf_node(level, leaf_node_id, data, target):
    node = LeafNode(n_samples=data.shape[0], level=level, node_id=leaf_node_id)

    for label in sorted(data[target].unique()):
        node.values[label] = data[data[target] == label].shape[0]
    node.function = max(node.values.items(), key=operator.itemgetter(1))[0]
    return node


def grow_with_stop(train_dataset, features, important_class, depth, target, beta, nodes, current_level,
                   weights, max_measure):
    """

    :param train_dataset: training set
    :param features: Feature column names
    :param bb_model: Black box model to explain
    :param important_class: Current class to overestimate
    :param depth: Max tree depth
    :param target: Target column name
    :param beta: Beta parameter for choosing F1 or F0.5 measure, default = 1
    :param nodes: List of nodes
    :param current_level: Current depth
    :param weights: class weights (deprecated)
    :param max_measure: Max F1 measure on the path
    :return: Trained Confident Decision Tree
    """
    split_feature, split_threshold, measure = confident_split(train_dataset, features, target, important_class,
                                                              beta, weights)

    if split_feature is None or measure < max_measure:
        node = create_leaf_node(level=current_level, leaf_node_id=len(nodes), data=train_dataset, target=target)
        nodes.append(node)
    else:
        max_measure = measure
        train_dataset_left = train_dataset[train_dataset[split_feature] <= split_threshold]
        train_dataset_right = train_dataset[train_dataset[split_feature] > split_threshold]
        if train_dataset_right.shape[0] == 0 or train_dataset_left.shape[0] == 0:
            node = create_leaf_node(level=current_level, leaf_node_id=len(nodes), data=train_dataset, target=target)
            nodes.append(node)
        else:
            node = InternalNode(level=current_level, n_samples=train_dataset.shape[0], node_id=len(nodes))
            for label in sorted(train_dataset[target].unique()):
                node.values[label] = train_dataset[train_dataset[target] == label].shape[0]
            node.feature = split_feature
            node.threshold = split_threshold
            node.depth = current_level
            nodes.append(node)

            if current_level < depth - 1:
                if train_dataset_left[target].nunique() > 1 and train_dataset_left.shape[0] >= 2:
                    node.child_left = grow_with_stop(train_dataset_left, features, important_class, depth,
                                                     target, beta, nodes, current_level+1, weights, max_measure)
                else:
                    leaf_node = create_leaf_node(level=current_level + 1, leaf_node_id=len(nodes),
                                                 data=train_dataset_left, target=target)
                    node.child_left = leaf_node
                    nodes.append(node.child_left)
                if train_dataset_right[target].nunique() > 1 and train_dataset_right.shape[0] >= 2:
                    node.child_right = grow_with_stop(train_dataset_right, features, important_class, depth,
                                                      target, beta, nodes, current_level + 1, weights, max_measure)
                else:
                    leaf_node = create_leaf_node(level=current_level + 1, leaf_node_id=len(nodes),
                                                 data=train_dataset_right, target=target)
                    node.child_right = leaf_node
                    nodes.append(node.child_right)
            elif current_level == depth - 1:
                leaf_node = create_leaf_node(level=current_level + 1, leaf_node_id=len(nodes), data=train_dataset_left,
                                             target=target)
                node.child_left = leaf_node
                nodes.append(node.child_left)
                leaf_node = create_leaf_node(level=current_level + 1, leaf_node_id=len(nodes), data=train_dataset_right,
                                             target=target)
                node.child_right = leaf_node
                nodes.append(node.child_right)

    return node


def confident_split(dataset, features, target, important_class, beta, weights):
    best_feature = None
    best_threshold = None
    best_measure = 0.0
    for f in features:
        values = dataset[f].unique()
        best_feature_measure = 0.0
        best_feature_threshold = None
        for value in values:
            measure = f_split(dataset, f, value, target, important_class, weights, beta)
            if measure > best_feature_measure:
                best_feature_measure = measure
                best_feature_threshold = value

        if best_feature_measure > best_measure:
            best_measure = best_feature_measure
            best_threshold = best_feature_threshold
            best_feature = f

    return best_feature, best_threshold, best_measure


def compute_confidence(tree, decision_path, x):
    confidence = 0.0

    for n_id in decision_path:
        for n in tree.nodes:
            if n.node_id == n_id:
                node = n
        if isinstance(node, InternalNode):
            if x[node.feature] <= node.threshold:
                confidence += max(node.child_left.values.values()) / node.child_left.n_samples
            else:
                confidence += max(node.child_right.values.values()) / node.child_right.n_samples
        else:
            confidence += max(node.values.values()) / node.n_samples

    return confidence/len(decision_path)


def f_split(data, feature, v, target, important_class, weights, beta):

    larger_per_class = {}
    smaller_per_class = {}

    for label in sorted(data[target].unique()):
        dataset = data[data[target] == label]
        c_larger = dataset[dataset[feature] > v].shape[0]
        c_smaller = dataset[dataset[feature] <= v].shape[0]
        larger_per_class[label] = c_larger
        smaller_per_class[label] = c_smaller
    
    if sum(smaller_per_class.values()) == 0 or sum(larger_per_class.values()) == 0:
        return 0.0
    if important_class not in smaller_per_class:
        class_smaller = 0.0
    else:
        class_smaller = smaller_per_class[important_class] / sum(smaller_per_class.values())
    if important_class not in larger_per_class:
        class_larger = 0.0
    else:
        class_larger = larger_per_class[important_class] / sum(larger_per_class.values())

    if class_smaller >= class_larger:
        measure = compute_f1(smaller_per_class, larger_per_class, important_class, weights, beta)
    else:
        measure = compute_f1(larger_per_class, smaller_per_class, important_class, weights, beta)

    return measure


def compute_f1(positives, negatives, main_class, w, beta):
    tp = 0.0
    tn = 0.0
    fn = 0.0
    fp = 0.0
    for key, value in positives.items():
        if key == main_class:
            tp += w[main_class] * value
        else:
            fp += value

    for key, value in negatives.items():
        if key == main_class:
            fn += w[main_class] * value
        else:
            tn += value

    precision = tp / (tp + fp + 0.000001)
    recall = tp / (tp + fn + 0.000001)
    
    return ((1 + beta ** 2) * precision * recall) / ((beta ** 2) * precision + recall + 0.000001)


def maxi_depth(node):
    if node is None or isinstance(node, LeafNode):
        return 0
    else:
        l_depth = maxi_depth(node.child_left)
        r_depth = maxi_depth(node.child_right)

        if l_depth > r_depth:
            return l_depth + 1
        else:
            return r_depth + 1


def compute_confidence_leaf(tree, decision_path, x):
    confidence = 0.0
    total = 0
    for n_id in decision_path:
        node = tree.nodes[n_id]
        if not isinstance(node, InternalNode):
            confidence = max(node.values.values())
            total = node.n_samples
    return confidence, total