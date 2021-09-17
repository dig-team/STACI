from math import floor, ceil
from sklearn.cluster import KMeans
from unic_clustering import *
from disc_utils import *
import numpy as np
from sklearn.metrics import silhouette_score


def discretization(data, number_of_intervals=0, max_percentage_error=None, bin_width=None, bin_size=None):
    models = [equal_width_intervals, equal_frequency_intervals, kmeans_clustering]
    clusters = {}

    if number_of_intervals > 1:
        return return_n_clusters(data, number_of_intervals)
    elif bin_width:
        n_clusters = int(1 / bin_width)
        if n_clusters > 1:
            return equal_width_intervals(data, n_clusters)
        else:
            raise ValueError("The bin width is too big. Bin width should be in range (0, 0.5)")
    elif bin_size:
        n_clusters = int(len(data)/bin_size)
        if n_clusters > 1:
            return equal_frequency_intervals(data, n_clusters)
        else:
            raise ValueError("The bin size is too big. Bin size should be in range [1, {})".format(int(len(data)/2)))

    elif max_percentage_error:
        if max_percentage_error <= 0.0 or max_percentage_error >= 1.0:
            raise ValueError('The maximum allowed error must take value from the range (0, 1)')
        clusters_unic = unic_algorithm(data)
        number_of_intervals = len(clusters_unic.keys())
        min_error = 1.0
        for i in range(number_of_intervals, int(sqrt(len(data)))):
            for model in models:
                candidate_clusters = wrapper(model, data, i)
                predicted = predict_cluster(data, candidate_clusters)
                error = evaluate_cluster(data, predicted)
                if error < min_error:
                    min_error = error
                    clusters = candidate_clusters
            if min_error < max_percentage_error:
                return clusters

        print("Reached maximum number of iterations. "
              "Returning the discretization for {} number of bins".format(int(sqrt(len(data)))))
        return clusters

    else:
        # User didn't provide any input. We use UNIC to determine the approximate number of intervals for
        # discretization
        clusters_unic = unic_algorithm(data)
        number_of_intervals = len(clusters_unic.keys())
        iterations = 10
        silhouette_max = -1.0
        i_max = 1
        m = "Unic"
        if len(clusters_unic.keys()) > 1:
            silhouette_unic = compute_silhouette_score(clusters_unic, data)
            if silhouette_unic > silhouette_max:
                silhouette_max = silhouette_unic
                i_max = len(clusters_unic.keys())
                clusters = clusters_unic

        for i in range(number_of_intervals, number_of_intervals + iterations + 1):
            if i > 1:
                for model in models:
                    candidate_clusters = wrapper(model, data, i)
                    silhouette_max, i_max, clusters = update_max_silhouette(candidate_clusters, data, silhouette_max,
                                                                            i_max, i, clusters)
                    if candidate_clusters == clusters:
                        m = model

        # print(str(m))
        return clusters


def return_n_clusters(data, n):
    models = [equal_width_intervals, equal_frequency_intervals, kmeans_clustering]
    best_clusters = {}
    best_silhouette = -1.0
    for model in models:
        clusters = wrapper(model, data, n)
        silhouette = compute_silhouette_score(clusters, data)
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_clusters = clusters

    return best_clusters


def wrapper(func, *args):
    return func(*args)


def equal_width_intervals(data, n_intervals):
    X = data.copy()
    max_value = max(X)
    min_value = min(X)

    width = (max_value - min_value) / n_intervals
    X.sort()

    intervals = {}
    bin_number = 0
    intervals[bin_number] = {}
    intervals[bin_number] = {'items': [], 'min': min_value + bin_number * width,
                             'max': min_value + (bin_number + 1) * width}

    for item in X:
        new_bin_number = floor((item - min_value) / width)
        if bin_number == n_intervals:
            bin_number = n_intervals - 1

        if bin_number in intervals and new_bin_number == bin_number:
            intervals[bin_number]['items'].append(item)
        else:
            bin_number += 1
            intervals[bin_number] = {'items': [], 'min': min_value + bin_number * width,
                                     'max': min_value + (bin_number + 1) * width}
            if intervals[bin_number]['min'] <= item < intervals[bin_number]['max']:
                intervals[bin_number]['items'].append(item)
            else:
                bin_number += 1
                intervals[bin_number] = {'items': [], 'min': min_value + bin_number * width,
                                         'max': min_value + (bin_number + 1) * width}
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
    intervals[bin_number] = {'items': []}
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
            intervals[bin_number] = {'items': []}
            intervals[bin_number]['items'].append(X[i])

    for key, value in intervals.items():
        value['median'] = median(value['items'])

    intervals = compute_edges(interval_dict=intervals, n_bins=len(intervals.keys()))
    return intervals


def kmeans_clustering(data, n_intervals):
    data = np.reshape(data, (-1, 1))
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
                break
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
