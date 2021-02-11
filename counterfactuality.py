from statistics import mean
from cf_nodes import *


def counterfactuality(model_to_explain, explainer, X, categorical_features):
    # probabilities before
    same_tree_prediction_change = []
    same_tree_proba_change = []
    for i in range(len(X)):
        instance = X.iloc[i]
        class_apriori = model_to_explain.predict([instance])[0]
        #print("class before: ", class_apriori)
        proba_apriori = max(model_to_explain.predict_proba([instance])[0])
        tree_class = explainer.trees[class_apriori].predict_single(instance)
        dpath = explainer.trees[class_apriori].decision_path(instance)

        if class_apriori == tree_class:
            pred_changed, p_changed = compute_aposteriori(instance, dpath, explainer.trees[class_apriori],
                                                          model_to_explain, class_apriori, proba_apriori,
                                                          categorical_features)
            #print(pred_changed, p_changed)
            same_tree_prediction_change.append(pred_changed)
            same_tree_proba_change.append(p_changed)

    same_tree_prediction_change_ratio = sum(same_tree_prediction_change)/len(same_tree_prediction_change)
    same_tree_proba_change_average = mean(same_tree_proba_change)
    return same_tree_prediction_change_ratio, same_tree_proba_change_average


def compute_aposteriori(x, decision_path, tree, model_to_explain, class_label, class_proba, categorical):
    temp = x.copy()
    prediction_change = 0
    proba_change = 0.0
    total_proba_change = 0.0
    new_proba = class_proba
    for i in range(len(decision_path)):
        feature_to_change = None
        if isinstance(tree.nodes[i], InternalNode):
            if not(tree.nodes[i].feature in categorical):
                #print(temp[tree.nodes[i].feature])
                temp[tree.nodes[i].feature] = tree.nodes[i].threshold
                #print(temp[tree.nodes[i].feature])
                #print("changed")
            else:
                if temp[tree.nodes[i].feature] == 0:
                    temp[tree.nodes[i].feature] = 1
                    start = tree.nodes[i].feature.split("_")[0]
                    for key, value in categorical.items():
                        if key.startswith(start) and key != tree.nodes[i].feature:
                            temp[key] = 0
                elif temp[tree.nodes[i].feature] == 1:
                    temp[tree.nodes[i].feature] = 0
                    start = tree.nodes[i].feature.split("_")[0]
                    for key, value in categorical.items():
                        if key.startswith(start) and key != tree.nodes[i].feature:
                            feature_to_change = key
                    if feature_to_change is not None:
                        temp[feature_to_change] = 1
        class_aposteriori = model_to_explain.predict([temp])[0]
        #print("Class after: ", class_aposteriori)
        proba_aposteriori = model_to_explain.predict_proba([temp])[0]
        #print("Proba change: ", new_proba - proba_aposteriori[class_label])
        total_proba_change += new_proba - proba_aposteriori[class_label]
        new_proba = proba_aposteriori[class_label]

        if class_label != class_aposteriori:
            prediction_change = 1

    return prediction_change, total_proba_change
