import random
from statistics import median, mean, stdev
from math import sqrt, exp
from statsmodels.stats.stattools import medcouple
from disc_utils import *


def unic_algorithm(data):

    a1 = random.choice(data)
    a2 = random.choice(data)
    a3 = random.choice(data)

    data_dict = {}
    for i in range(len(data)):
        data_dict[i] = data[i]

    distances1 = compute_distances(a1, data_dict)
    distances2 = compute_distances(a2, data_dict)
    distances3 = compute_distances(a3, data_dict)

    clusters1 = partitioning(distances1, data_dict)
    clusters2 = partitioning(distances2, data_dict)
    clusters3 = partitioning(distances3, data_dict)
    neighbours1 = find_neighbours(clusters1, clusters2)
    clusters = find_neighbours(neighbours1, clusters3)

    updated_clusters = update_cluster_metrics(clusters)

    return updated_clusters


def compute_distances(ref_point, points):
    distances = {}

    for key, value in points.items():
        distances[key] = value - ref_point

    return distances


def compute_differences(distances_dict):
    differences = {}
    diff = []
    previous = 0
    for key, value in distances_dict.items():
        differences[key] = distances_dict[key] - previous
        diff.append(differences[key])
        previous = distances_dict[key]

    return differences, diff


def compute_3rd_quartile(data):

    med = median(data)
    upper_half = [item for item in data if item > med]
    lower_half = [item for item in data if item < med]

    if len(upper_half) == 0:
        upper_half = [item for item in data if item >= med]
    elif len(lower_half) == 0:
        lower_half = [item for item in data if item <= med]

    return median(upper_half), median(upper_half) - median(lower_half)


def partitioning(distances, points_dict):
    clusters = {}
    sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
    differences, diff = compute_differences(sorted_distances)

    n = len(distances.keys())
    p = int(sqrt(n))

    counter = 0
    dist_diff_subset = []
    for key, value in differences.items():
        if p-1 <= counter <= n - p - 1:
            dist_diff_subset.append(value)
        counter += 1
    mean_subset = mean(dist_diff_subset)
    sd_subset = stdev(dist_diff_subset)
    clean_subset = [item for item in dist_diff_subset if item > mean_subset + sd_subset]

    q3_clean, iqr_clean = compute_3rd_quartile(clean_subset)
    mc_clean = medcouple(clean_subset)
    if mc_clean < 0.6:
        ref_dist_diff = q3_clean + 1.5 * (exp(3 * mc_clean)) * iqr_clean
    else:
        ref_dist_diff = q3_clean + 1.5 * iqr_clean
    current_cluster = 0
    countdown_value = p/2
    countdown = countdown_value + 1

    clusters[current_cluster] = {}
    clusters[current_cluster]['items'] = []
    point_counter = 0
    for key, value in differences.items():
        if point_counter <= p:
            clusters[current_cluster]['items'].append(points_dict[key])
            point_counter += 1

    point_counter = 0
    for key, value in differences.items():
        if point_counter > p:
            if value > ref_dist_diff:
                if p + point_counter < n:
                    if p + point_counter < n - p:
                        temp_max = compute_max_of_a_slice(differences, point_counter+1, point_counter + p)
                    else:
                        temp_max = compute_max_of_a_slice(differences, point_counter + 1, n - p)

                    if value > temp_max and countdown <= 0:
                        current_cluster += 1
                        clusters[current_cluster] = {}
                        clusters[current_cluster]['items'] = []
                        countdown = countdown_value + 1

            clusters[current_cluster]['items'].append(points_dict[key])
            countdown -= 1
        point_counter += 1

    return clusters


def compute_max_of_a_slice(data_dict, start, stop):

    data_slice = []

    counter = 0
    for key, value in data_dict.items():
        if start <= counter <= stop:
            data_slice.append(value)
        counter += 1

    return max(data_slice)


def find_neighbours(clusters1, clusters2):

    neighbours = {}
    clusters = 0
    for key1, value1 in clusters1.items():
        for key2, value2 in clusters2.items():
            neighbours[clusters] = {}
            neighbours[clusters]['items'] = []
            for item1 in value1['items']:
                if item1 in value2['items']:
                    neighbours[clusters]['items'].append(item1)

            clusters += 1

    keys_to_delete = []
    for key, value in neighbours.items():
        if len(value['items']) == 0:
            keys_to_delete.append(key)

    for k in keys_to_delete:
        del neighbours[k]

    return neighbours


def update_cluster_metrics(clusters):
    for key, value in clusters.items():
        value['min'] = min(value['items'])
        value['max'] = max(value['items'])
        value['median'] = median(value['items'])

    sorted_clusters = dict(sorted(clusters.items(), key=lambda item: item[1]['min']))

    new_clusters = {}
    cluster_id = 0
    for key, value in sorted_clusters.items():
        new_clusters[cluster_id] = value
        cluster_id += 1

    n_bins = len(new_clusters.keys())
    final_clusters = compute_edges(new_clusters, n_bins)

    return final_clusters
