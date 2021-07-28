import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from math import floor, ceil
from statistics import median
from sklearn.cluster import KMeans


def discretization(x, number_of_intervals=0):
    intervals = {}

    if number_of_intervals == 0:
        pass

    return intervals


def equal_width_intervals(data, n_intervals):
    max_value = max(data)
    min_value = min(data)

    width = (max_value - min_value) / n_intervals
    print(width)
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
    print(bin_size)
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
            intervals[prediction]['items'].append(int(data[i][0]))
        else:
            intervals[prediction] = {}
            intervals[prediction]['items'] = [int(data[i][0])]

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


def compute_edges(interval_dict, n_bins):

    for key, value in interval_dict.items():
        if key == 0:
            value['min'] = min(value['items'])
            value['max'] = (min(interval_dict[key + 1]['items']) + max(value['items'])) / 2
        elif key < n_bins - 1:
            value['min'] = interval_dict[key - 1]['max']
            value['max'] = (min(interval_dict[key + 1]['items']) + max(value['items'])) / 2
        else:
            value['min'] = interval_dict[key - 1]['max']
            value['max'] = max(value['items'])

    return interval_dict
