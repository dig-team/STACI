from staci import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from statistics import mean
from sklearn.neural_network import MLPRegressor
from disc_utils import evaluate_cluster
import time

datasets = ["auto", "bike", "housing", "servo", "concrete"]

depths = [3, 4, 5, 6]
start_time = time.time()
print("Start time: ", start_time)
for dataset in datasets:
    df = pd.read_csv("../datasets/regression/" + dataset + ".csv", header=0)
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

    trials = [i for i in range(10)]
    fidelity = {}
    coverage = {}
    confidence = {}
    wapa_complex = []
    av_generality = {}
    average_length = {}
    n_nodes = {}
    counter = {}
    m_depth = {}
    counter_proba = {}
    number_of_clusters = {}
    mse_fidelity = {}
    mse_coverage = {}

    for trial in trials:
        trainDf, testDf, y_train, y_test = train_test_split(X, y, test_size=0.1)
        # black_box = RandomForestRegressor(n_estimators=1000)
        black_box = MLPRegressor(solver='adam', hidden_layer_sizes=(500,), max_iter=1000)
        black_box.fit(trainDf, y_train)
        y_pred = black_box.predict(trainDf)
        y_pred_test = black_box.predict(testDf)
        wapa_complex.append(evaluate_cluster(y_test, y_pred_test))
        y_pred_df = pd.Series((v for v in y_pred), name="target", index=trainDf.index)
        data = data_preparation(trainDf, y_pred_df, attrs, 'target')
        data_to_discretize = data['target'].tolist()
        data = data.rename(columns={'target': 'numerical_target'})
        intervals = discretization(data_to_discretize, max_percentage_error=0.1)
        clusters = intervals
        print(clusters.keys())
        print("Discretization done!")
        print("--- %s seconds ---" % (time.time() - start_time))
        new_target1 = convert_to_labels(data_to_discretize, intervals)
        new_target = pd.Series((v for v in new_target1), name="target", index=data.index)
        data['target'] = new_target

        labels_count = {}
        labels_predict = convert_to_labels(y_pred, clusters)
        for item in labels_predict:
            if item in labels_count:
                labels_count[item] += 1
            else:
                labels_count[item] = 1

        for depth in depths:
            if depth not in fidelity:
                fidelity[depth] = []
                coverage[depth] = []
                confidence[depth] = []
                av_generality[depth] = []
                average_length[depth] = []
                n_nodes[depth] = []
                counter[depth] = []
                m_depth[depth] = []
                counter_proba[depth] = []
                number_of_clusters[depth] = []
                mse_coverage[depth] = []
                mse_fidelity[depth] = []

            explainer = STACISurrogates(max_depth=depth, regression=True)
            explainer.clusters = clusters
            explainer.fit(data, features=attrs, target='target')
            print("Training done!")
            print("--- %s seconds ---" % (time.time() - start_time))
            number_of_clusters[depth].append(len(labels_count.keys()))

            max_nodes = 0
            maximum_depth = 0
            for key, dtree in explainer.trees.items():
                if len(dtree.nodes) > max_nodes:
                    max_nodes = len(dtree.nodes)
                tree_depth = maxi_depth(dtree.nodes[0])
                if tree_depth > maximum_depth:
                    maximum_depth = tree_depth
            """
            for t, value in explainer.trees.items():
                print("Tree: ", t)
                for node in value.nodes:
                    if isinstance(node, InternalNode):
                        print("Node id: {}, Node feature: {}, Threshold: {}, Values: {}, Level: {}"
                              .format(node.node_id, node.feature, node.threshold, node.values, node.depth))
                    else:
                        print("Leaf Node id: {}, Values: {}, Level: {}".format(node.node_id, node.values, node.depth))
            """

            exp_predict = explainer.predict(testDf, black_box)
            conf_predict, confidence_sample, leaf_values, explanation_length = explainer.confidence_predict(testDf,
                                                                                                            labels_count)
            print("Prediction done!")
            print("--- %s seconds ---" % (time.time() - start_time))
            fidelity_sample = evaluate_cluster(y_pred_test, conf_predict)
            coverage_sample = evaluate_cluster(y_pred_test, exp_predict)
            mse_fidelity_sample = mean_squared_error(y_pred_test, conf_predict)
            mse_coverage_sample = mean_squared_error(y_pred_test, exp_predict)
            print(fidelity_sample, coverage_sample)

            if fidelity_sample < coverage_sample:
                print("*********************************************************")
                print("Wrong clusters!")
                print(explainer.clusters)
                print(y_pred_test)
                print(conf_predict)
                print(exp_predict)
                print("*********************************************************")

            fidelity[depth].append(fidelity_sample)
            coverage[depth].append(coverage_sample)
            confidence[depth].append(mean(confidence_sample))
            n_nodes[depth].append(max_nodes)
            m_depth[depth].append(maximum_depth)
            av_generality[depth].append(mean(leaf_values))
            average_length[depth].append(mean(explanation_length))
            mse_fidelity[depth].append(mse_fidelity_sample)
            mse_coverage[depth].append(mse_coverage_sample)
    with open("nn_results_max_error_" + dataset + ".txt", 'w') as resultfile:
        for depth in depths:
            resultfile.write("=====================================================================\n")
            resultfile.write("Dataset, Depth: " + dataset + ", " + str(depth) + "\n")
            resultfile.write("Fidelity: " + str(fidelity[depth]) + "\n")
            resultfile.write("Average Fidelity: " + str(mean(fidelity[depth])) + "\n")
            resultfile.write("Coverage: " + str(coverage[depth]) + "\n")
            resultfile.write("Average Coverage: " + str(mean(coverage[depth])) + "\n")
            resultfile.write("MSE Average Fidelity: " + str(mean(mse_fidelity[depth])) + "\n")
            resultfile.write("MSE Fidelity: " + str(mse_fidelity[depth]) + "\n")
            resultfile.write("MSE Average Coverage: " + str(mean(mse_coverage[depth])) + "\n")
            resultfile.write("MSE Coverage: " + str(mse_coverage[depth]) + "\n")
            resultfile.write("Average length: " + str(mean(average_length[depth])) + "\n")
            resultfile.write("Maximal length: " + str(mean(m_depth[depth])) + "\n")
            resultfile.write("Confidence: " + str(confidence[depth]) + "\n")
            resultfile.write("Average Confidence: " + str(mean(confidence[depth])) + "\n")
            resultfile.write("Generality: " + str(av_generality[depth]) + "\n")
            resultfile.write("Average Generality: " + str(mean(av_generality[depth])) + "\n")
            resultfile.write("Average number of clusters: " + str(mean(number_of_clusters[depth])) + "\n")
            resultfile.write("Number of clusters: " + str(number_of_clusters[depth]) + "\n")
            resultfile.write("=====================================================================\n")
