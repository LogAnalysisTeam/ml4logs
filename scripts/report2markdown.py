import argparse

from collections import Counter, defaultdict, OrderedDict
import json
import numpy as np
import os
import pathlib
from pathlib import Path
import pandas as pd
import typing
from typing import Dict, List
from sklearn.base import TransformerMixin

import ml4logs
from ml4logs.data.hdfs import load_data_as_dict, load_labels

def import_result_files(result_dir):
    metrics = []
    for res_file in sorted(Path(result_dir).glob("*.json")):
        print(res_file)

        with open(res_file, "r") as f:
            data = json.load(f, object_hook=OrderedDict)
            for method, results in data["metrics"].items():
                rec = OrderedDict([("method", method), ("preprocess", res_file.stem)])
                rec.update(results)
                metrics.append(rec)
    df =  pd.DataFrame(metrics)
    return df

def markdown_table_string(result_df, method2str, pre2str):
    df = result_df[["method", "preprocess", "precision", "recall", "f1", "mcc"]].copy()
    df.rename(columns={"method": "Method", "preprocess": "Preprocess", "f1": "F1", "mcc": "MCC", 
        "precision": "Precision", "recall": "Recall"}, inplace=True)
    df.sort_values("F1", ascending=False, inplace=True)

    df['Method']= df['Method'].map(lambda m: method2str.get(m, m))
    df['Preprocess']= df['Preprocess'].map(lambda p: pre2str.get(p, p))
    df['Precision'] = df['Precision'].map("{:.3f}".format)
    df['Recall'] = df['Recall'].map("{:.3f}".format)
    df['F1'] = df['F1'].map("{:.3f}".format)
    df['MCC'] = df['MCC'].map("{:.3f}".format)
    
    def mark_best(col):
        vals = pd.to_numeric(df[col]).values
        best = np.max(vals)
        mask = (vals == best)
        df[col][mask] = df[col][mask].map("**{}**".format) 
        
    mark_best('Precision')
    mark_best('Recall')
    mark_best('F1')
    mark_best('MCC')
    
    return df.to_markdown(showindex=False)

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produce Markdown table from  LAD experiment result JSON files.')
    parser.add_argument('result_dir', type=dir_path, nargs=1,
                        help='result directory, e.g., "results/HDFS1"')

    args = parser.parse_args()

    df = import_result_files(args.result_dir[0])
    method2str = {
        "pca": "PCA",
        "decision_tree": "Decision Tree",
        "logistic_regression": "Logistic Regression",
        "one_class_svm": "One Class SVM",
        "lof_sklearn": "Local Outlier Factor (`sklearn`)",
        "linear_svc": "Linear SVC",
        "isolation_forest_sklearn": "Isolation Forest (`sklearn`)",
        "lstm_classifier_m2o": "LSTM M2O",
        "lstm_classifier_m2m": "LSTM M2M",
    }

    pre2str = {
        "ibm_drain-unsupervised-loglizer": "Drain3",
        "ibm_drain-loglizer": "Drain3",
        "ibm_drain-unsupervised-loglizer": "Drain3",
        "fasttext_timedeltas_minmax_blockmax-unsupervised-loglizer": "fastText block-max",
        "fasttext_timedeltas_minmax_blockmax-loglizer": "fastText block-max",
        "fasttext_timedeltas_minmax-seq2label": "fastText",
        "fasttext_timedeltas_minmax-seq2label-m2m": "fastText",
    }
    print(markdown_table_string(df, method2str, pre2str))