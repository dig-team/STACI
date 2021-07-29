from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error


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


def predict_cluster(data, clusters):

    prediction = []
    max_key = max(clusters.keys())
    for item in data:
        for key, value in clusters.items():
            if key < max_key:
                if value['min'] <= item < value['max']:
                    prediction.append(value['median'])
            else:
                if value['min'] <= item <= value['max']:
                    prediction.append(value['median'])

    return prediction


def evaluate_cluster(data, predicted):

    mae = mean_absolute_error(data, predicted)
    mape = mean_absolute_percentage_error(data, predicted)

    return mae, mape
