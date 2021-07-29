from math import floor, ceil
from sklearn.cluster import KMeans
from unic_clustering import *
from disc_utils import *
import numpy as np


def discretization(data, number_of_intervals=0):
    intervals = {}

    clusters_unic = unic_algorithm(data)

    if number_of_intervals == 0:
        number_of_intervals = len(clusters_unic.keys())
    unic_predict = predict_cluster(data, clusters_unic)
    unic_mae, unic_mape = evaluate_cluster(data, unic_predict)
    print("UNIC: ", unic_mae, unic_mape)
    clusters_ew = equal_width_intervals(data, number_of_intervals)
    ew_predict = predict_cluster(data, clusters_ew)
    ew_mae, ew_mape = evaluate_cluster(data, ew_predict)
    print("EW: ", ew_mae, ew_mape)
    clusters_ef = equal_frequency_intervals(data, number_of_intervals)
    ef_predict = predict_cluster(data, clusters_ef)
    ef_mae, ef_mape = evaluate_cluster(data, ef_predict)
    print("EF: ", ef_mae, ef_mape)
    clusters_kmeans = kmeans_clustering(np.reshape(data, (-1, 1)), number_of_intervals)
    kmeans_predict = predict_cluster(data, clusters_kmeans)
    kmeans_mae, kmeans_mape = evaluate_cluster(data, kmeans_predict)
    print("Kmeans: ", kmeans_mae, kmeans_mape)

    return intervals


def equal_width_intervals(data, n_intervals):
    max_value = max(data)
    min_value = min(data)

    width = (max_value - min_value) / n_intervals
    data.sort()

    intervals = {}

    for item in data:
        bin_number = floor((item-min_value)/width)
        if bin_number == n_intervals:
            bin_number = n_intervals - 1

        if bin_number in intervals:
            intervals[bin_number]['items'].append(item)
        else:
            intervals[bin_number] = {}
            intervals[bin_number]['items'] = []
            intervals[bin_number]['items'].append(item)
            intervals[bin_number]['min'] = min_value + bin_number * width
            intervals[bin_number]['max'] = min_value + (bin_number + 1) * width

    for key, value in intervals.items():
        value['median'] = median(value['items'])

    return intervals


def equal_frequency_intervals(data, n_intervals):
    bin_size = ceil(len(data) / n_intervals)
    data.sort()

    intervals = {}

    for i in range(len(data)):
        bin_number = floor(i/bin_size)
        if bin_number > n_intervals - 1:
            bin_number = n_intervals - 1

        if bin_number in intervals:
            intervals[bin_number]['items'].append(data[i])
        else:
            intervals[bin_number] = {}
            intervals[bin_number]['items'] = []
            intervals[bin_number]['items'].append(data[i])

    for key, value in intervals.items():
        value['median'] = median(value['items'])

    intervals = compute_edges(interval_dict=intervals, n_bins=n_intervals)

    return intervals


def kmeans_clustering(data, n_intervals):

    cls = KMeans(n_clusters=n_intervals, init='k-means++', n_init=10).fit(data)

    intervals = {}
    medians = []
    for i in range(len(data)):
        prediction = int(cls.predict([data[i]])[0])
        if prediction in intervals:
            intervals[prediction]['items'].append(float(data[i][0]))
        else:
            intervals[prediction] = {}
            intervals[prediction]['items'] = [float(data[i][0])]

    for key, value in intervals.items():
        value['median'] = median(value['items'])
        medians.append(median(value['items']))

    intervals2 = {}

    medians.sort()

    clusters = 0
    for item in medians:
        for key, value in intervals.items():
            if value['median'] == item:
                intervals2[clusters] = value
                clusters += 1

    intervals = compute_edges(interval_dict=intervals2, n_bins=n_intervals)

    return intervals
