# from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


def compute_edges(interval_dict, n_bins):

    if n_bins == 1:
        for key, value in interval_dict.items():
            if key == 0:
                value['min'] = float('-inf')
                value['max'] = float('inf')

    else:
        for key, value in interval_dict.items():
            if key == 0:
                value['min'] = float('-inf')
                value['max'] = (min(interval_dict[key + 1]['items']) + max(value['items'])) / 2
            elif key < n_bins - 1:
                value['min'] = interval_dict[key - 1]['max']
                value['max'] = (min(interval_dict[key + 1]['items']) + max(value['items'])) / 2
            else:
                value['min'] = interval_dict[key - 1]['max']
                value['max'] = float('inf')

    return interval_dict


#
def predict_cluster(data, clusters):
    prediction = []
    for item in data:
        for key, value in clusters.items():
            if value['min'] <= item < value['max']:
                prediction.append(value['median'])
                break

    return prediction


def label_by_cluster(data, clusters):
    labels = []

    for item in data:
        for key, value in clusters.items():
            if value['min'] <= item < value['max']:
                labels.append(key)
                break

    return labels


def convert_to_labels(data, clusters):
    prediction = []

    for item in data:
        for key, value in clusters.items():
            if item in value['items']:
                prediction.append(key)
                break

    return prediction


def evaluate_cluster(data, predicted):
    abs_sum = 0.0
    data = list(data)
    for i in range(len(data)):
        abs_sum += abs(data[i] - predicted[i])

    wape = (abs_sum / sum(data))
    """
    mae = mean_absolute_error(data, predicted)
    mape = mean_absolute_percentage_error(data, predicted)
    mse = mean_squared_error(data, predicted)
    """

    return wape
