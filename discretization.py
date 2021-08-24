from math import floor, ceil
from sklearn.cluster import KMeans
from unic_clustering import *
from disc_utils import *
import numpy as np
from sklearn.metrics import silhouette_score
import csv


def discretization(data, number_of_intervals=0):
    iterations = 10
    clusters_unic = unic_algorithm(data)
    silhouette_max = 0.0
    i_max = 1
    clusters = {}
    if number_of_intervals == 0:
        number_of_intervals = len(clusters_unic.keys())

    if len(clusters_unic.keys()) > 1:
        silhouette_unic = compute_silhouette_score(clusters_unic, data)
        if silhouette_unic > silhouette_max:
            silhouette_max = silhouette_unic
            i_max = len(clusters_unic.keys())
            clusters = clusters_unic

    for i in range(number_of_intervals, number_of_intervals + iterations + 1):
        clusters_ew = equal_width_intervals(data, i)
        clusters_ef = equal_frequency_intervals(data, i)
        clusters_kmeans = kmeans_clustering(np.reshape(data, (-1, 1)), i)
        if i > 1:
            silhouette_max, i_max, clusters = update_max_silhouette(clusters_ew, data, silhouette_max, i_max, i, clusters)
            silhouette_max, i_max, clusters = update_max_silhouette(clusters_ef, data, silhouette_max, i_max, i, clusters)
            silhouette_max, i_max, clusters = update_max_silhouette(clusters_kmeans, data, silhouette_max, i_max, i, clusters)

    return clusters


def equal_width_intervals(data, n_intervals):
    X = data.copy()
    max_value = max(X)
    min_value = min(X)

    width = (max_value - min_value) / n_intervals
    X.sort()

    intervals = {}
    bin_number = 0
    intervals[bin_number] = {}
    intervals[bin_number]['items'] = []
    intervals[bin_number]['min'] = min_value + bin_number * width
    intervals[bin_number]['max'] = min_value + (bin_number + 1) * width

    for item in X:
        new_bin_number = floor((item - min_value) / width)
        if bin_number == n_intervals:
            bin_number = n_intervals - 1

        if bin_number in intervals and new_bin_number == bin_number:
            intervals[bin_number]['items'].append(item)
        else:
            bin_number += 1
            intervals[bin_number] = {}
            intervals[bin_number]['items'] = []
            intervals[bin_number]['min'] = min_value + bin_number * width
            intervals[bin_number]['max'] = min_value + (bin_number + 1) * width
            if intervals[bin_number]['min'] <= item < intervals[bin_number]['max']:
                intervals[bin_number]['items'].append(item)
            else:
                bin_number += 1
                intervals[bin_number] = {}
                intervals[bin_number]['items'] = []
                intervals[bin_number]['min'] = min_value + bin_number * width
                intervals[bin_number]['max'] = min_value + (bin_number + 1) * width
                intervals[bin_number]['items'].append(item)

    for key, value in intervals.items():
        if len(value['items']) > 0:
            value['median'] = median(value['items'])
        else:
            value['median'] = (intervals[bin_number]['max'] + intervals[bin_number]['min']) / 2

    return intervals


def equal_frequency_intervals(data, n_intervals):
    bin_size = ceil(len(data) / n_intervals)
    X = data.copy()
    X.sort()

    intervals = {}
    bin_number = 0
    intervals[bin_number] = {}
    intervals[bin_number]['items'] = []
    new_bin_number = 0
    for i in range(len(X)):
        if bin_number > n_intervals - 1:
            bin_number = n_intervals - 1

        if i > 0 and X[i - 1] != X[i]:
            new_bin_number = floor(i / bin_size)

        if bin_number in intervals and bin_number == new_bin_number:
            intervals[bin_number]['items'].append(X[i])
        elif X[i - 1] == X[i]:
            intervals[bin_number]['items'].append(X[i])
        else:
            bin_number += 1
            intervals[bin_number] = {}
            intervals[bin_number]['items'] = []
            intervals[bin_number]['items'].append(X[i])

    for key, value in intervals.items():
        value['median'] = median(value['items'])

    intervals = compute_edges(interval_dict=intervals, n_bins=len(intervals.keys()))
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


def return_cluster_labels(clusterer, X):
    labels = []
    for item in X:
        found = False
        for key, value in clusterer.items():
            if item in value["items"]:
                labels.append(key)
                found = True
        if not found:
            print(item)

    return labels


def compute_silhouette_score(cls, data):
    cluster_labels = return_cluster_labels(cls, data)

    return silhouette_score(np.reshape(data, (-1, 1)), cluster_labels)


def update_max_silhouette(clusters, data, previous, optimal_n_of_clusters, k, previous_clusters):
    silhouette = compute_silhouette_score(clusters, data)
    if previous is None:
        previous = silhouette
        optimal_n_of_clusters = k
        previous_clusters = clusters
    elif silhouette > previous:
        previous = silhouette
        optimal_n_of_clusters = k
        previous_clusters = clusters

    return previous, optimal_n_of_clusters, previous_clusters

