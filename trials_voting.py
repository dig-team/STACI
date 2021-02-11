from c2xai import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from statistics import mean
from counterfactuality import counterfactuality

#datasets = ["diabetes", "heart", "voting", "dermatology", "wine"]
datasets = ["adult"]#"adult", "sick", "hypothyroid", "breast32"]

depth = 4
for dataset in datasets:
    df = pd.read_csv("./datasets/" + dataset + ".csv", header=0)
    y = df["target"]
    X_raw = df.drop("target", axis=1)
    nominal_features = []
    numerical_features = []
    if dataset == "voting":
        depth = 3
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
    beta = [1]
    trials = [0,1,2,3,4,5,6,7,8,9]
    fidelity = {}
    coverage = {}
    confidence = {}
    train_confidence = {}
    accuracy_complex = []
    n_nodes = {}
    counter = {}
    counter_proba = {}
    labels = y.unique()
    labels.sort()
    print(labels)
    for trial in trials:
        trainDf, testDf, y_train, y_test = train_test_split(X, y, test_size=0.1)
        # black_box = RandomForestClassifier(n_estimators=1000)
        black_box = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(500,))
        black_box.fit(trainDf, y_train)
        y_pred = black_box.predict(trainDf)
        y_pred_labels = np.unique(y_pred)
        print(y_pred_labels)
        while not (len(labels) == len(y_pred_labels)):
            trainDf, testDf, y_train, y_test = train_test_split(X, y, test_size=0.1)
            # black_box = RandomForestClassifier(n_estimators=1000)
            black_box = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(500,))
            black_box.fit(trainDf, y_train)
            y_pred = black_box.predict(trainDf)
            y_pred_labels = np.unique(y_pred)
            print(y_pred_labels)

        y_pred_test = black_box.predict(testDf)

        print("Training accuracy: ", accuracy_score(y_train, y_pred))
        print("Test accuracy: ", accuracy_score(y_test, y_pred_test))
        accuracy_complex.append(accuracy_score(y_test, y_pred_test))
        y_pred_df = pd.Series((v for v in y_pred), name="target", index=trainDf.index)
        for b in beta:
            print(dataset, depth, "nn", "f1", b)
            explainer = CFSurrogates(max_depth=depth, beta=b, weighted=False)
            explainer.fit(trainDf, y_pred_df, black_box, features=attrs, target='target')
            max_nodes = 0
            for key, dtree in explainer.trees.items():
                print("Class " + str(key) + " number of nodes: ", len(dtree.nodes))
                if len(dtree.nodes) > max_nodes:
                    max_nodes = len(dtree.nodes)

            for t, value in explainer.trees.items():
                print("Tree: ", t)
                for node in value.nodes:
                    if isinstance(node, InternalNode):
                        print("Node id: {}, Node feature: {}, Threshold: {}, Values: {}, Level: {}"
                              .format(node.node_id, node.feature, node.threshold, node.values, node.depth))
                    else:
                        print("Leaf Node id: {}, Values: {}, Level: {}".format(node.node_id, node.values, node.depth))
            exp_predict = explainer.predict(testDf, black_box)
            conf_predict, confidence_sample = explainer.confidence_predict(testDf)
            train_conf_predict, train_confidence_sample = explainer.confidence_predict(trainDf)
            fidelity_sample = accuracy_score(y_pred_test, conf_predict)
            coverage_sample = accuracy_score(y_pred_test, exp_predict)
            stpcr, stpca = counterfactuality(black_box, explainer, testDf, nominal_dummy_features)

            if b in fidelity:
                fidelity[b].append(fidelity_sample)
                coverage[b].append(coverage_sample)
                confidence[b].append(mean(confidence_sample))
                counter[b].append(stpcr)
                counter_proba[b].append(stpca)
                n_nodes[b].append(max_nodes)
                train_confidence[b].append(mean(train_confidence_sample))
            else:
                fidelity[b] = []
                coverage[b] = []
                confidence[b] = []
                counter[b] = []
                counter_proba[b] = []
                n_nodes[b] = []
                train_confidence[b] = []
                fidelity[b].append(fidelity_sample)
                coverage[b].append(coverage_sample)
                confidence[b].append(mean(confidence_sample))
                counter[b].append(stpcr)
                counter_proba[b].append(stpca)
                n_nodes[b].append(max_nodes)
                train_confidence[b].append(mean(train_confidence_sample))
            print(trial, dataset)
            print(fidelity_sample)
            print(coverage_sample)
            print(mean(train_confidence_sample))
            print(mean(confidence_sample))
            print(stpcr)
            print(stpca)
    print("Fidelity: ", fidelity)
    print("Coverage: ", coverage)
    print("Confidence: ", confidence)
    print("Train confidence: ", train_confidence)
    print("Counter predict: ", mean(counter[beta[0]]))
    print("Counter proba: ", mean(counter_proba[beta[0]]))
    average_confidence = mean(confidence[beta[0]])
    output_file = open("./" + dataset + "_nn_test" + str(depth) + ".txt", 'a')
    for b in beta:
        output_file.writelines(
            "Dataset: {}, \nTree depth: {},\nResults: \nBeta: {} \nCoverage: {} \nAccuracy_complex: {}\n"
            "Avg Confidence Fidelity: {}\nNumber of nodes in boosted tree: {}\nAverage Confidence: {}\nAverage train Confidence: {}\n"
            "Prediction change: {}\nProba change: {}\n"
            .format(dataset, depth, b, mean(coverage[b]), mean(accuracy_complex),
                    mean(fidelity[b]),
                    mean(n_nodes[b]), mean(confidence[b]), mean(train_confidence[b]),
                    mean(counter[b]), mean(counter_proba[b])))

    output_file.close()





