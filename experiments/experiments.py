from staci import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from statistics import mean
from counterfactuality import counterfactuality

datasets = ["voting", "diabetes", "heart", "breast32", "wine", "sick", "hypothyroid", "dermatology", "adult"]

depth = 4

for dataset in datasets:
    df = pd.read_csv("../datasets/" + dataset + ".csv", header=0)
    y = df["target"]
    X_raw = df.drop("target", axis=1)
    nominal_features = []
    numerical_features = []
    if dataset == "voting":
        depth = 4
        for col in X_raw.columns:
            nominal_features.append(col)

    if dataset == "diabetes" or dataset == "breast32" or dataset == "wine":
        depth = 4
        for col in X_raw.columns:
            numerical_features.append(col)

    if dataset == "heart":
        nominal_features = ["sex", "cp", "restecg", "slope", "thal"]
        numerical_features = []
        for col in X_raw.columns:
            if col not in nominal_features:
                numerical_features.append(col)
        depth = 4
    if dataset == "sick":
        nominal_features = ["sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "sick",
                            "pregnant",
                            "thyroid_surgery", "I131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium",
                            "goitre", "tumor", "hypopituitary", "psych", "TSH_measured", "T3_measured", "TT4_measured",
                            "T4U_measured", "FTI_measured", "TBG_measured", "referral_source"]
        depth = 4
        for col in X_raw.columns:
            if col not in nominal_features:
                numerical_features.append(col)
    if dataset == "hypothyroid":
        nominal_features = ["sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "thyroid_surgery",
                            "query_hypothyroid", "query_hyperthyroid", "pregnant", "sick", "tumor", "lithium", "goitre",
                            "TSH_measured", "T3_measured", "TT4_measured", "T4U_measured", "FTI_measured",
                            "TBG_measured"]
        depth = 4
        for col in X_raw.columns:
            if col not in nominal_features:
                numerical_features.append(col)

    if dataset == "adult":
        nominal_features = ["workclass", "education", "marital-status", "occupation", "relationship", "sex"]
        depth = 4
        for col in X_raw.columns:
            if col not in nominal_features:
                numerical_features.append(col)

    if dataset == "dermatology":
        nominal_features = ['family_history']
        depth = 4
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
    accuracy_complex = []
    av_generality = []
    average_length = []
    n_nodes = []
    counter = []
    m_depth = []
    counter_proba = []
    labels = y.unique()
    labels.sort()
    for trial in trials:
        print("Dataset: ", dataset)
        trainDf, testDf, y_train, y_test = train_test_split(X, y, test_size=0.1)
        black_box = RandomForestClassifier(n_estimators=1000)
        # black_box = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(500,))
        black_box.fit(trainDf, y_train)
        y_pred = black_box.predict(trainDf)
        y_pred_labels = np.unique(y_pred)
        while not (len(labels) == len(y_pred_labels)):
            trainDf, testDf, y_train, y_test = train_test_split(X, y, test_size=0.1)
            black_box = RandomForestClassifier(n_estimators=1000)
            # black_box = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(500,))
            black_box.fit(trainDf, y_train)
            y_pred = black_box.predict(trainDf)
            y_pred_labels = np.unique(y_pred)

        labels_count = {}
        for item in y_pred:
            if item in labels_count:
                labels_count[item] += 1
            else:
                labels_count[item] = 1
        y_pred_test = black_box.predict(testDf)
        accuracy_complex.append(accuracy_score(y_test, y_pred_test))
        y_pred_df = pd.Series((v for v in y_pred), name="target", index=trainDf.index)

        explainer = STACISurrogates(max_depth=depth)
        explainer.fit(trainDf, y_pred_df, black_box, features=attrs, target='target')
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
        conf_predict, confidence_sample, leaf_values, explanation_length = explainer.confidence_predict(testDf,
                                                                                                        labels_count)
        fidelity_sample = accuracy_score(y_pred_test, conf_predict)
        coverage_sample = accuracy_score(y_pred_test, exp_predict)
        stpcr, stpca = counterfactuality(black_box, explainer, testDf, nominal_dummy_features)

        fidelity.append(fidelity_sample)
        coverage.append(coverage_sample)
        confidence.append(mean(confidence_sample))
        counter.append(stpcr)
        counter_proba.append(stpca)
        n_nodes.append(max_nodes)
        m_depth.append(maximum_depth)
        av_generality.append(mean(leaf_values))
        average_length.append(mean(explanation_length))

    print("Coverage: ", coverage)
    print("Average length: ", mean(average_length))
    print("Maximal length: ", mean(m_depth))
    print("Confidence: ", confidence)
    print("Generality: ", mean(av_generality))
