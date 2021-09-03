from staci import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from statistics import mean
from sklearn.neural_network import MLPRegressor
from disc_utils import evaluate_cluster

datasets = ["auto", "electrical", "bike", "servo", "housing", "concrete", "superconduct"]

depth = 4

for dataset in datasets:
    df = pd.read_csv("../datasets/" + dataset + ".csv", header=0)
    y = df["target"]
    X_raw = df.drop("target", axis=1)
    nominal_features = []
    numerical_features = []

    if dataset in ["auto", "electrical", "concrete", "superconduct"]:
        nominal_features = []
        for col in X_raw.columns:
            numerical_features.append(col)

    if dataset == "housing":
        nominal_features = ["chas"]
        for col in X_raw.columns:
            if col not in nominal_features:
                numerical_features.append(col)

    if dataset == "servo":
        numerical_features = []
        for col in X_raw.columns:
            nominal_features.append(col)

    if dataset == "bike":
        nominal_features = ["Seasons", "Holiday", "Functioning_day"]
        for col in X_raw.columns:
            if col not in nominal_features:
                numerical_features.append(col)

    X = pd.get_dummies(data=X_raw, columns=nominal_features)
    nominal_dummy_features = {}
    for col in X.columns:
        if col not in numerical_features:
            unique_values = X[col].unique().tolist()
            nominal_dummy_features[col] = unique_values

    attrs = []
    for column in X.columns:
        attrs.append(column)

    trials = [i for i in range(20)]
    fidelity = []
    coverage = []
    confidence = []
    wapa_complex = []
    av_generality = []
    average_length = []
    n_nodes = []
    counter = []
    m_depth = []
    counter_proba = []

    for trial in trials:
        print("Dataset: ", dataset)
        trainDf, testDf, y_train, y_test = train_test_split(X, y, test_size=0.1)
        black_box = RandomForestRegressor(n_estimators=1000)
        # black_box = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(500,))
        black_box.fit(trainDf, y_train)
        y_pred = black_box.predict(trainDf)

        y_pred_test = black_box.predict(testDf)
        wapa_complex.append(evaluate_cluster(y_test, y_pred_test))
        y_pred_df = pd.Series((v for v in y_pred), name="target", index=trainDf.index)

        explainer = STACISurrogates(max_depth=depth, regression=True)
        explainer.fit(trainDf, y_pred_df, features=attrs, target='target')
        labels_count = 0  # TODO

        max_nodes = 0
        maximum_depth = 0
        for key, dtree in explainer.trees.items():
            if len(dtree.nodes) > max_nodes:
                max_nodes = len(dtree.nodes)
            tree_depth = maxi_depth(dtree.nodes[0])
            if tree_depth > maximum_depth:
                maximum_depth = tree_depth

        for t, value in explainer.trees.items():
            print("Tree: ", t)
            for node in value.nodes:
                if isinstance(node, InternalNode):
                    print("Node id: {}, Node feature: {}, Threshold: {}, Values: {}, Level: {}"
                          .format(node.node_id, node.feature, node.threshold, node.values, node.depth))
                else:
                    print("Leaf Node id: {}, Values: {}, Level: {}".format(node.node_id, node.values, node.depth))

        exp_predict = explainer.predict(testDf, black_box)
        conf_predict, confidence_sample, leaf_values, explanation_length = explainer.confidence_predict(testDf, labels_count)

        fidelity_sample = accuracy_score(y_pred_test, conf_predict)
        coverage_sample = accuracy_score(y_pred_test, exp_predict)

        fidelity.append(fidelity_sample)
        coverage.append(coverage_sample)
        confidence.append(mean(confidence_sample))
        n_nodes.append(max_nodes)
        m_depth.append(maximum_depth)
        av_generality.append(mean(leaf_values))
        average_length.append(mean(explanation_length))

    print("Coverage: ", coverage)
    print("Average length: ", mean(average_length))
    print("Maximal length: ", mean(m_depth))
    print("Confidence: ", confidence)
    print("Generality: ", mean(av_generality))
