import numpy as np
from pathlib import Path
import json
import torch
import time

from datasets import CustomMinMaxScaler
from utils import load_pickle_file, find_optimal_threshold, classify
from metrics import metrics_report
from autoencoder_tcnn import AETCN

config = {
    "hyperparameters": {
        "batch_size": 8,
        "dropout": 0.3238513794447296,
        "epochs": 4,
        "input_shape": 100,
        "kernel_size": 3,
        "layers": [
            [
                142
            ],
            1246,
            [
                100  # output_shape == input_shape
            ]
        ],
        "learning_rate": 0.0016378937069540646,
        "window": 45
    },
    # not used currently
    "model_path": "../../models/aetcn/4f5f4682-1ca5-400a-a340-6243716690c0.pt",
    "threshold": 0.00331703620031476
}


def aetcn_experiment(args):
    assert args.X_train_dataset is not None
    assert args.y_train_dataset is not None
    assert args.X_val_dataset is not None
    assert args.y_val_dataset is not None
    assert args.X_test_dataset is not None
    assert args.y_test_dataset is not None
    assert args.output_parent_folder is not None
    assert args.embedding_model_name is not None

    NAME = f'AETCN_using_{args.embedding_model_name}'
    print(NAME)
    print(args)

    output_path = Path(args.output_parent_folder) / NAME.replace(' ', '_')

    X_train_raw = load_pickle_file(args.X_train_dataset)
    y_train = np.load(args.y_train_dataset)
    X_val_raw = load_pickle_file(args.X_val_dataset)
    y_val = np.load(args.y_val_dataset)
    X_test_raw = load_pickle_file(args.X_test_dataset)
    y_test = np.load(args.y_test_dataset)

    sc = CustomMinMaxScaler()
    X_train = sc.fit_transform(X_train_raw)
    X_val = sc.transform(X_val_raw)
    X_test = sc.transform(X_test_raw)

    model = AETCN()
    model.set_params(**config['hyperparameters'])

    model.fit(X_train[y_train == 0])  # train only on Normal data (label 0)
    y_val_pred_raw = model.predict(X_val)

    val_theta, val_f1 = find_optimal_threshold(y_val, y_val_pred_raw)
    y_val_pred = classify(y_val_pred_raw, val_theta)
    print("Metrics on val using theta estimated from val")
    val_report = metrics_report(y_val, y_val_pred)

    y_test_pred_raw = model.predict(X_test)
    y_test_pred_using_val_theta = classify(y_test_pred_raw, val_theta)
    print("Metrics on test using theta estimated from val")
    test_report_using_val = metrics_report(y_test, y_test_pred_using_val_theta)

    test_theta, test_f1 = find_optimal_threshold(y_test, y_test_pred_raw)
    y_test_pred_using_test_theta = classify(y_test_pred_raw, test_theta)
    print("Metrics on test using theta estimated from test -- Hyperparameters should not be estimated from test")
    test_report_using_test = metrics_report(y_test, y_test_pred_using_test_theta)

    run_info = {
        'run_name': NAME,
        'hyperparameters' : config['hyperparameters'],
        'theta_val': val_theta,
        'report_val': val_report,
        'report_test_using_val': test_report_using_val,
        'theta_test': test_theta,
        'report_test_using_test': test_report_using_test
    }

    output_path.mkdir(parents=True, exist_ok=True)

    INFO_FILE_PATH = output_path / 'run_info.json'
    with INFO_FILE_PATH.open(mode='w') as f:
        f.write(json.dumps(run_info, indent=4))

    MODEL_PATH = output_path / 'model.pt'
    torch.save(model, str(MODEL_PATH))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Anomaly detection using AutoEncoder from Temporal Convolution Network")
    parser.add_argument('--output-parent-folder', default=None, type=str)
    parser.add_argument('--embedding-model-name', default=None, type=str, help="Name of the model that created the embeddings")
    parser.add_argument("--X-train-dataset", default=None, type=str)
    parser.add_argument("--y-train-dataset", default=None, type=str)
    parser.add_argument("--X-val-dataset", default=None, type=str)
    parser.add_argument("--y-val-dataset", default=None, type=str)
    parser.add_argument("--X-test-dataset", default=None, type=str)
    parser.add_argument("--y-test-dataset", default=None, type=str)

    args = parser.parse_args()
    aetcn_experiment(args)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f'Total time taken: {end - start}s')