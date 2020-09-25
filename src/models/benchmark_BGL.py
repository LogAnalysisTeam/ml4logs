#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from os.path import join as pjoin
import sys

import pandas as pd
from loglizer.models import *
from loglizer import dataloader, preprocessing

run_models = [
    'PCA',
    'InvariantsMiner',
    'IsolationForest',
    'LR',
    'SVM',
    'DecisionTree'
    'LogClustering',
    'DeepLog',
    ]

log_path = sys.argv[1]  # 'Drain_result/'
log_file = sys.argv[2]  # 'BGL_100k.log_structured.csv'
result_dir = sys.argv[3]

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    logger.info("loading BGL dataset")
    (x_tr, y_train), (x_te, y_test) = dataloader.load_BGL(pjoin(log_path, log_file),
                                                          save_path=log_path,
                                                          window='sliding',
                                                          time_interval=1,
                                                          stepping_size=1,
                                                          train_ratio=0.8)

    benchmark_results = []
    for _model in run_models:
        logger.info('Evaluating {} on BGL:'.format(_model))
        if _model == 'PCA':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf',
                                                      normalization='zero-mean')
            model = PCA()
            model.fit(x_train)

        elif _model == 'InvariantsMiner':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr)
            model = InvariantsMiner(epsilon=0.5)
            model.fit(x_train)

        elif _model == 'LogClustering':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = LogClustering(max_dist=0.3, anomaly_threshold=0.3)
            model.fit(x_train[y_train == 0, :])  # Use only normal samples for training

        elif _model == 'IsolationForest':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr)
            model = IsolationForest(random_state=2019, max_samples=0.9999, contamination=0.03,
                                    n_jobs=4)
            model.fit(x_train)

        elif _model == 'LR':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = LR()
            model.fit(x_train, y_train)

        elif _model == 'SVM':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = SVM()
            model.fit(x_train, y_train)

        elif _model == 'DecisionTree':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = DecisionTree()
            model.fit(x_train, y_train)

        elif _model == "DeepLog":
            batch_size = 32
            hidden_size = 32
            num_directions = 2
            topk = 5
            window_size = 10
            epoches = 5
            num_workers = 2
            device = 0

            feature_extractor = preprocessing.Vectorizer()
            train_dataset = feature_extractor.fit_transform(*dataloader.slice_hdfs(x_tr, y_train, window_size))
            test_dataset = feature_extractor.transform(*dataloader.slice_hdfs(x_te, y_test, window_size))

            train_loader = preprocessing.Iterator(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers).iter
            test_loader = preprocessing.Iterator(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers).iter

            model = DeepLog(num_labels=feature_extractor.num_labels, hidden_size=hidden_size, num_directions=num_directions, topk=topk, device=device)
            model.fit(train_loader, epoches)

            logger.info('Train accuracy:')
            metrics = model.evaluate(train_loader)
            benchmark_results.append([_model + '-train', metrics['precision'], metrics['recall'], metrics['f1']])
            logger.info('Test accuracy:')
            metrics = model.evaluate(test_loader)
            benchmark_results.append([_model + '-test', metrics['precision'], metrics['recall'], metrics['f1']])
            continue

        x_test = feature_extractor.transform(x_te)
        logger.info('Train accuracy:')
        precision, recall, f1 = model.evaluate(x_train, y_train)
        benchmark_results.append([_model + '-train', precision, recall, f1])
        logger.info('Test accuracy:')
        precision, recall, f1 = model.evaluate(x_test, y_test)
        benchmark_results.append([_model + '-test', precision, recall, f1])
 
    pd.DataFrame(benchmark_results, columns=['Model', 'Precision', 'Recall', 'F1']) \
        .to_csv(pjoin(result_dir, 'benchmark_result.csv'), index=False)
