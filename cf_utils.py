import pandas as pd
import numpy as np
from copy import deepcopy
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

    print(weights)

    return weights


def create_leaf_node(level, id, data, target):
    node = LeafNode(n_samples=data.shape[0], level=level, node_id=id)

    for label in sorted(data[target].unique()):
        node.values[label] = data[data[target] == label].shape[0]
    node.function = max(node.values.items(), key=operator.itemgetter(1))[0]
    return node


def grow(train_dataset, features, bb_model, important_class, depth, target, beta, nodes, current_level, weights):
    split_feature, split_threshold, measure = counterfactual_split(train_dataset, bb_model, features, target,
                                                             important_class, beta, weights)

    if split_feature is None:
        node = create_leaf_node(level=current_level, id=len(nodes), data=train_dataset, target=target)
        nodes.append(node)
    else:
        train_dataset_left = train_dataset[train_dataset[split_feature] <= split_threshold]
        # print(train_dataset_left.shape[0])
        train_dataset_right = train_dataset[train_dataset[split_feature] > split_threshold]
        # print(train_dataset_right.shape[0])
        if train_dataset_right.shape[0] == 0 or train_dataset_left.shape[0] == 0:
            node = create_leaf_node(level=current_level, id=len(nodes), data=train_dataset, target=target)
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
                    node.child_left = grow(train_dataset_left, features, bb_model, important_class, depth,
                                       target, beta, nodes, current_level+1, weights)
                else:
                    leaf_node = create_leaf_node(level=current_level + 1, id=len(nodes), data=train_dataset_left,
                                             target=target)
                    node.child_left = leaf_node
                    nodes.append(node.child_left)
                if train_dataset_right[target].nunique() > 1 and train_dataset_right.shape[0] >= 2:
                    node.child_right = grow(train_dataset_right, features, bb_model, important_class,
                                        depth, target, beta, nodes, current_level + 1, weights)
                else:
                    leaf_node = create_leaf_node(level=current_level + 1, id=len(nodes), data=train_dataset_right,
                                             target=target)
                    node.child_right = leaf_node
                    nodes.append(node.child_right)
            elif current_level == depth - 1:
                print("Reached max depth: ", current_level)
                leaf_node = create_leaf_node(level=current_level + 1, id=len(nodes), data=train_dataset_left, target=target)
                node.child_left = leaf_node
                nodes.append(node.child_left)
                leaf_node = create_leaf_node(level=current_level + 1, id=len(nodes), data=train_dataset_right, target=target)
                node.child_right = leaf_node
                nodes.append(node.child_right)

    return node


def counterfactual_split(dataset, bb_model, features, target, important_class, beta, weights):
    best_feature = None
    best_threshold = None
    best_measure = 0.0
    max_conf_measure = 0.0
    for f in features:
        values = dataset[f].unique()
        best_feature_measure = 0.0
        best_conf_measure = 0.0
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


def conf_gen_split(dataset, bb_model, features, target, important_class, beta, weights):
    best_feature = None
    best_threshold = None
    best_measure = 0.0
    measures = {}
    beta_conf = {}
    for f in features:
        values = dataset[f].unique()
        for value in values:
            confidence, generality = f_split(dataset, f, value, target, important_class, weights, beta)
            measures[(f, value)] = (confidence, generality)
            if confidence >= beta:
                beta_conf[(f, value)] = (confidence, generality)

    if bool(beta_conf):
        max_gen = 0.0
        for key, value in beta_conf.items():
            if value[1] > max_gen:
                max_gen = value[1]
                best_feature = key[0]
                best_threshold = key[1]
                best_measure = value[0]
    else:
        max_conf = 0.0
        for key, value in measures.items():
            if value[0] > max_conf:
                max_conf = value[0]
                best_feature = key[0]
                best_threshold = key[1]
                best_measure = value[0]

    return best_feature, best_threshold, best_measure


def compute_gini(data, feature, v, target):

    total_number = data.shape[0]
    n_larger = []
    n_smaller = []
    count_larger = data[data[feature] > v].shape[0]
    count_smaller = data[data[feature] <= v].shape[0]
    for label in sorted(data[target].unique()):
        dataset = data[data[target] == label]
        c_larger = dataset[dataset[feature] > v].shape[0]
        c_smaller = dataset[dataset[feature] <= v].shape[0]
        n_larger.append((c_larger/(count_larger + 1)) ** 2)
        n_smaller.append((c_smaller/(count_smaller + 1)) ** 2)

    return (count_larger/total_number) * (1 - sum(n_larger)) + (count_smaller/total_number) * (1 - sum(n_smaller))


def compute_confidence_measure(data, feature, v, target, important_class, weights):

    count_larger = data[data[feature] > v].shape[0]
    count_smaller = data[data[feature] <= v].shape[0]
    dataset = data[data[target] == important_class]
    c_larger = dataset[dataset[feature] > v].shape[0]
    c_smaller = dataset[dataset[feature] <= v].shape[0]
    proba_larger = c_larger/(count_larger + 1)
    proba_smaller = c_smaller/(count_smaller+1)

    if proba_smaller != 0 or proba_larger != 0:
        if proba_smaller > proba_smaller:
            return proba_smaller, True
        else:
            return proba_larger, False
    else:
        measure = compute_weighted_gini(data, feature, v, target, weights, important_class)
        return measure


def compute_confidence(tree, decision_path, x):
    confidence = 0.0

    for n_id in decision_path:
        node = tree.nodes[n_id]
        if isinstance(node, InternalNode):
            if x[node.feature] <= node.threshold:
                confidence += max(node.child_left.values.values()) / node.child_left.n_samples
            else:
                confidence += max(node.child_right.values.values()) / node.child_right.n_samples
        else:
            confidence += max(node.values.values()) / node.n_samples

    return confidence/len(decision_path)


def compute_weighted_gini(data, feature, v, target, weights, important_class):
    total_number = 0
    count_larger = 0
    count_smaller = 0
    df_larger = data[data[feature] > v]
    df_smaller = data[data[feature] <= v]
    for class_label in sorted(data[target].unique()):
        df = data[data[target] == class_label]
        num_smaller = df_smaller[df_smaller[target] == class_label].shape[0]
        num_larger = df_larger[df_larger[target] == class_label].shape[0]
        if class_label == important_class:
            total_number += weights[class_label] * df.shape[0]
            count_larger += weights[class_label] * num_larger
            count_smaller += weights[class_label] * num_smaller
        else:
            total_number += df.shape[0]
            count_larger += num_larger
            count_smaller += num_smaller
    n_larger = []
    n_smaller = []
    for label in sorted(data[target].unique()):
        dataset = data[data[target] == label]
        if label == important_class:
            w = weights[label]
        else:
            w = 1
        c_larger = w * dataset[dataset[feature] > v].shape[0]
        c_smaller = w * dataset[dataset[feature] <= v].shape[0]
        n_larger.append((c_larger/(count_larger + 1)) ** 2)
        n_smaller.append((c_smaller/(count_smaller + 1)) ** 2)

    return (count_larger/total_number) * (1 - sum(n_larger)) + (count_smaller/total_number) * (1 - sum(n_smaller))


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
