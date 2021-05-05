## Installation
1. Clone the source:
`https://github.com/LogAnalysisTeam/ml4logs`

2. Activate your virtual environment (conda, venv).

3. Either install the package as usual: 

`python setup.py install`

or in development regime:

`python setup.py develop`

## Usage

Various pipelines are run using batches in `scripts/`. We suggest to run the scripts via `Makefile`:

`make COMMAND_NAME`

The scripts support SLURM cluster batch scheduler. Set `ML4LOGS_SHELL` environment variable to `sbatch` in case you perform the experiments on the cluster. See [RCI Quick Start](docs/RCI_quick_start.md) for full details on how to setup development environment.

If  `init_environment.sh` script exists in the project root directory, it is `source`d (via `bash` `source` command) prior running any batch in `scripts/`. Use it to set up your virtual environment, scheduler modules, etc.

### Run Benchmark on HDFS1 (100k lines)

- `make hdfs1_100k_data`
- wait until finish
- `make hdfs1_100k_preprocess`
- wait
- `make hdfs1_100k_train_test`

### Run Benchmark on HDFS1

- `make hdfs1_data`
- wait
- `make hdfs1_preprocess`
- wait
- `make hdfs1_train_test`

## Results

The following table (generated using [script](scripts/report2markdown.py)) shows the current LAD method leaderboard for the HDFS1 dataset. The methods are sorted by decreasing F1 score.

### Unsupervised/Semi-Supervised Methods
| Method                           | Preprocess         | Precision   | Recall    | F1        | MCC       |
|:---------------------------------|:-------------------|:------------|:----------|:----------|:----------|
| PCA                              | Drain3             | 0.849       | 0.809     | **0.828** | **0.824** |
| Isolation Forest (`sklearn`)     | Drain3             | 0.808       | 0.800     | 0.804     | 0.798     |
| Local Outlier Factor (`sklearn`) | Drain3             | 0.429       | **0.928** | 0.587     | 0.616     |
| Isolation Forest (`sklearn`)     | fastText block-max | **0.989**   | 0.364     | 0.532     | 0.594     |
| PCA                              | fastText block-max | 0.380       | 0.384     | 0.382     | 0.363     |
| Local Outlier Factor (`sklearn`) | fastText block-max | 0.258       | 0.014     | 0.027     | 0.055     |

### Supervised Methods
| Method              | Preprocess         | Precision   | Recall    | F1        | MCC       |
|:--------------------|:-------------------|:------------|:----------|:----------|:----------|
| Decision Tree       | Drain3             | 0.997       | **0.999** | **0.998** | **0.998** |
| Logistic Regression | Drain3             | 0.980       | 0.995     | 0.988     | 0.987     |
| LSTM M2O            | fastText           | 0.992       | 0.471     | 0.639     | 0.678     |
| Decision Tree       | fastText block-max | 0.614       | 0.634     | 0.624     | 0.612     |
| Logistic Regression | fastText block-max | 0.911       | 0.420     | 0.575     | 0.612     |
| Linear SVC          | fastText block-max | 0.948       | 0.387     | 0.550     | 0.599     |
| Linear SVC          | Drain3             | **1.000**   | 0.230     | 0.375     | 0.475     |
| LSTM M2M            | fastText           | 0.874       | 0.111     | 0.197     | 0.309     |

**Notes:**
- Currently only LOF and IF methods for Drain3-preprocessed data have meta-parameters tuned (using grid or random search). We found the meta-parameter tunning extremely important. The results for other combinations of methods and preprocessing pipelines will follow soon...
- All experiments above included time-deltas merged with the rest of features.
- The features differ based on a selected preprocessing pipeline:
  - **Drain3**: Log keys are extracted getting per-block BOWs which is in turn weighted using TF-IDF. While we have currenlty best results for Drain3, the big disadvantage is, that the fixed categorical distribution over the log keys does not allow log lines based on yet unseen templates to be processed.
  - **fastText**: Block loglines are represented as a sequence of 100-dimensional fastText embbeddings.
  - **fastText block-max**: Tue same 100-dimensional fastText embbeddings aggregated to a single 100-dimensional vector using max-pooling.

## Scripts and Configuration Files

- Each script executes corresponds to a single pipeline config (see `configs/` directory)
- Config describes a sequential pipeline of actions which is applied to data


### `data`

- Downloads archive.
- Extracts archive.
- Prepares the dataset:
  - **TODO: ADD DETAILS HERE**
  - Time deltas are computed. Time deltas measure the time differences between successive log lines.

### `drain_preprocess`

- Parses log keys (log templates) using [IBM/Drain3](https://github.com/IBM/Drain3).
- Aggregates log lines by blocks, which correspond to level at which anomaly labels are given.

### `fasttext_preprocess`

- Trains the [fastText](https://fasttext.cc/) model.
- Gets embeddings for all log lines.
- Concatenates the embeddings with the time deltas.
- Aggregates per-log line embeddings to per-block ones using selected method (sum, average, min, max).

### `drain_loglizer`

Trains and tests models which are specified by [loglizer](https://github.com/logpai/loglizer) on Drain-parsed dataset. These are:
  - Logistic regression
  - Decision tree
  - Linear SVC
  - LOF
  - One class SVM
  - Isolation forest

### `fasttext_loglizer`

Trains and tests [loglizer](https://github.com/logpai/loglizer) specified models for aggregated fastText embeddings:
  - Logistic regression
  - Decision tree
  - Linear SVC
  - LOF
  - One class SVM
  - Isolation forest
  - PCA

### `fasttext_seq2seq`

- Trains and tests a sequential model as defined in [[1]](#1).
- Predicts the following log line embedding based on a history of log line embeddings.
- Uses LSTM based Torch model.
- Computes the threshold on a train dataset (assuming 5% logs are anomalies).
- Tests different thresholds and saves the statistics.

## Results
**TODO put result tables here**

## Data Files Description

### Block-Level Labeled Datasets (e.g., HDFS)

```
N - Number of log lines
B - Number of blocks (e.g. blk_ in HDFS)
E - Number of event ids (e.g. extracted by drain)
F - Embedding dimension (e.g. fasttext)
```

```
data
├── interim
│   └── {DATASET_NAME}
│       ├── blocks.npy                  (N, )       Block ids
│       ├── fasttext-timedeltas.npy     (N, F + 1)  Fasttext embeddings with timedeltas
│       ├── fasttext.npy                (N, F)      Fasttext embeddings
│       ├── ibm_drain-eventids.npy      (N, )       Event ids
│       ├── ibm_drain-templates.csv     (E, )       Event ids, their templates and occurrences
│       ├── labels.npy                  (B, )       Labels (1 stands for anomaly, 0 for normal)
│       ├── logs.txt                                Raw logs
│       └── timedeltas.npy              (N, )       Timedeltas
├── processed
│   └── {DATASET_NAME}
│       ├── fasttext-average.npz        (B, F + 1)  Fasttext embeddings with timedeltas aggregated by blocks
│       └── ibm_drain.npz               (B, E)      Count vectors
└── raw
    └── {DATASET_NAME}
        ├── {ARCHIVE}.tar.gz
        └── Dataset specific files
```

## References
<a id="1">[1]</a> 
M. Souček, ["Log Anomaly Detection"](https://dspace.cvut.cz/handle/10467/90271), master thesis, Czech Technical University in Prague, 2020.

