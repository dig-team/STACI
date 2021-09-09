# from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


def compute_edges(interval_dict, n_bins):

    if n_bins == 1:
        for key, value in interval_dict.items():
            if key == 0:
                value['min'] = min(value['items'])
                value['max'] = max(value['items'])

    else:
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


def predict_cluster(data, clusters):
    prediction = []
    max_key = max(clusters.keys())
    for item in data:
        for key, value in clusters.items():
            if key < max_key:
                if key == 0:
                    if item < value['max']:
                        prediction.append(value['median'])
                elif value['min'] <= item < value['max']:
                    prediction.append(value['median'])
            else:
                if item >= value['min']:
                    prediction.append(value['median'])

    return prediction


def convert_to_labels(data, clusters):
    prediction = []

    for item in data:
        for key, value in clusters.items():
            if item in value['items']:
                prediction.append(key)

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
