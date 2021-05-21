#
# Original work at: https://github.com/LogAnalysisTeam/methods4logfiles/blob/main/src/models/utils.py
#
#
#
#

from time import time_ns
from pathlib import Path
from typing import Callable, Iterable, Dict, List, Union
import json
import os
import pickle
import numpy as np
from scipy import stats

from anomaly_detection.metrics import get_metrics

SEED = 160121
np.random.seed(SEED)


def time_decorator(function: Callable):
    def wrapper(*arg, **kw):
        t1 = time_ns()
        ret = function(*arg, **kw)
        t2 = time_ns()
        return ret, (t2 - t1) * 1e-9

    return wrapper


def save_experiment(data: Iterable, file_path: str):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)


def load_experiment(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)


def create_experiment_report(metrics: Dict, hyperparameters: Dict, theta: float = None, file_path: str = None) -> Dict:
    ret = {
        'metrics': metrics,
        'hyperparameters': hyperparameters,
    }

    if theta:
        ret['threshold'] = theta
    if file_path:
        ret['model_path'] = file_path
    return ret


def create_model_path(dir_path: str, unique_name: str) -> str:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return os.path.join(dir_path, f'{unique_name}.pt')


def create_checkpoint(data: Iterable, file_path: str):
    save_experiment(data, file_path)


def load_pickle_file(file_path: str) -> Union[List, Dict]:
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def find_optimal_threshold(y_true: np.array, y_pred: np.array) -> tuple:
    ret = {}
    for th in set(y_pred[y_true == 1]):
        tmp = classify(y_pred, th)
        f1 = get_metrics(y_true, tmp)['f1_score']
        ret[th] = f1
    return max(ret.items(), key=lambda x: x[1])


def classify(y_pred: np.array, theta: float) -> np.array:
    ret = np.zeros(shape=y_pred.shape)
    ret[y_pred >= theta] = 1
    return ret


def get_encoder_size(layers: List):
    idx = 0
    while idx < len(layers) - 1 and layers[idx] < layers[idx + 1]:  # the sign is set by shape of channels
        idx += 1
    return idx + 1


def get_all_divisors(input_dim: int) -> List:
    # return sorted array of all divisors of `input_dim` in O(2 * sqrt(n)) time
    # a naive solution would take 0(n + n * log(n)) time
    divisors = []
    for i in range(1, int(np.sqrt(input_dim))):
        if input_dim % i == 0:
            divisors.append(i)
    for i in range(int(np.sqrt(input_dim)), 0, -1):
        if input_dim % i == 0:
            divisors.append(input_dim // i)
    return divisors


def get_number_of_items_within_range(divisors: List, lower: int, upper: int) -> int:
    ret = 0
    for i in divisors:
        if is_val_in_range(i, lower, upper):
            ret += 1
    return ret


def is_val_in_range(val: int, lower: int, upper: int) -> bool:
    return lower <= val <= upper


def get_normal_dist(divisors: List) -> List:
    if len(divisors) == 2:
        return [0.5, 0.5]

    x = np.asarray(divisors)
    x_upper, x_lower = x + 0.5, x - 0.5
    prob = stats.norm.cdf(x_upper, scale=5, loc=20) - stats.norm.cdf(x_lower, scale=5, loc=20)
    dist = prob / prob.sum()
    return dist.tolist()


def generate_layer_settings(input_dim: int, size: int) -> List:
    ret = []
    for i in range(size):
        layers = []

        n_encoder = np.random.randint(1, 4)
        layers_encoder = np.random.randint(50, 501, size=n_encoder)
        layers_encoder.sort(kind='mergesort')
        layers.extend(layers_encoder.tolist())  # ascending

        n_decoder = np.random.randint(0, 3)  # one layer is already included in the architecture itself
        layers_decoder = np.random.randint(50, layers[-1], size=n_decoder)
        layers_decoder.sort(kind='mergesort')
        layers.extend(layers_decoder.tolist()[::-1])  # descending

        ret.append(layers)
    return ret


def get_min_window_size(kernel: int, maxpool: int, n_encoder_layers: int) -> int:
    # time complexity might be improved with binary search O(log(n)) instead of O(n)
    for input_dim in range(1, 500):
        output_dim = get_window_size(input_dim, kernel, maxpool, n_encoder_layers)

        if output_dim > 0:
            return input_dim


def get_window_size(input_dim: int, kernel: int, maxpool: int, n_encoder_layers: int) -> int:
    output_dim = input_dim
    for _ in range(n_encoder_layers):
        output_dim = output_dim - kernel + 1  # Conv1d
        output_dim //= maxpool  # MaxPool1d
    return output_dim


def get_1d_window_size(encoder_kernels: List, layers: List, get_number_of_encoder_layers: Callable) -> List:
    maxpool = 2  # fixed also in the PyTorch model

    windows = []
    for i, kernel in enumerate(encoder_kernels):
        n_encoder_layers = get_number_of_encoder_layers(layers[i])
        min_window_size = get_min_window_size(kernel, maxpool, n_encoder_layers)
        windows.append(np.random.randint(min_window_size, min_window_size + 32))
    return windows


def get_2d_window_size(encoder_kernels: List, layers: List) -> List:
    maxpool = 2  # fixed also in the PyTorch model

    windows = []
    for i, kernel in enumerate(encoder_kernels):
        n_encoder_layers = get_encoder_size(layers[i])
        x_min_window_size = get_min_window_size(kernel[0], maxpool, n_encoder_layers)
        y_min_window_size = get_min_window_size(kernel[1], maxpool, n_encoder_layers)

        if x_min_window_size > 100:  # embeddings dimension
            raise AssertionError('Kernel needs greater embeddings dimension!')

        windows.append(np.random.randint(y_min_window_size, y_min_window_size + 32))
    return windows


def get_2d_kernels(x_choice: List, y_choice: List, n_experiments: int) -> List:
    x = np.random.choice(x_choice, size=n_experiments).tolist()
    y = np.random.choice(y_choice, size=n_experiments).tolist()
    return list(zip(x, y))


def get_encoder_heads(layers: List) -> List:
    ret = []
    for config in layers:
        n_encoder_layers = min(get_encoder_size(config), 2)
        divisors = get_all_divisors(config[n_encoder_layers - 1])
        ret.append(int(np.random.choice(divisors, p=get_normal_dist(divisors))))
    return ret


def get_decoder_heads(layers: List) -> List:
    ret = []
    for config in layers:
        n_decoder_layers = len(config) - get_encoder_size(config)

        if n_decoder_layers == 0:
            ret.append(None)
        else:
            divisors = get_all_divisors(config[-1])
            ret.append(int(np.random.choice(divisors, p=get_normal_dist(divisors))))
    return ret


def get_bottleneck_dim(layers: List) -> List:
    ret = []
    for config in layers:
        n_encoder_layers = get_encoder_size(config)

        n_channels = config[n_encoder_layers - 1]
        ret.append(int(np.random.randint(1, n_channels + 1)))
    return ret


def convert_predictions(y_pred: np.array) -> np.array:
    # LocalOutlierFactor and IsolationForest returns: 1 inlier, -1 outlier
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    return y_pred


def print_report(experiment_reports: Dict):
    print('+--------------------+-----------+-----------+-----------+')
    print('| Model              | Precision | Recall    | F1 Score  |')
    print('+--------------------+-----------+-----------+-----------+')
    for model_name, report in experiment_reports.items():
        metrics = report['test_metrics']
        n_spaces = 20 - len(model_name) - 1
        print(f'| {model_name}{" " * n_spaces}| {metrics["precision"]:.5f}   | {metrics["precision"]:.5f}   '
              f'| {metrics["f1_score"]:.5f}   |')
        print('+--------------------+-----------+-----------+-----------+')

