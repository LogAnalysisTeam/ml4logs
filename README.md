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

`make {COMMAND_NAME}`

The scripts support SLURM cluster batch scheduler. Set `ML4LOGS_SHELL` environment variable to `sbatch` in case you perform the experiments on the cluster.

If  `init_environment.sh` script exists in the project root directory, it is sourced (i.e., using `bash` `source` command) prior running any batch in `scripts/`. Use it to set up your virtual environment, scheduler modules, etc.

### Run benchmark on HDFS1 (100k lines)

- `make hdfs1_100k_data`
- wait
- `make hdfs1_100k_preprocess`
- wait
- `make hdfs1_100k_train_test`

### Run benchmark on HDFS1

- `make hdfs1_data`
- wait
- `make hdfs1_preprocess`
- wait
- `make hdfs1_train_test`

## Description of scripts and configs

- Each script executes only one config
- Config describes pipeline of actions which are applied to our data

### Scripts/configs

data

- Download archive
- Extract archive
- Prepare dataset

drain_preprocess

- Parse using IBM/Drain3
- Aggregate by block using count vector

fasttext_preprocess

- Train fasttext model
- Get embeddings for all log lines
- Merge with timedeltas
- Aggregate by block using selected method (sum, average, min, max)

drain_loglizer

- Train and test models which are used by loglizer on drain parsed dataset
  - Logistic regression
  - Decision tree
  - Linear SVC
  - LOF
  - One class SVM
  - Isolation forest

fasttext_loglizer

- Train and test models which are used by loglizer on aggregated fasttext embeddings
  - Logistic regression
  - Decision tree
  - Linear SVC
  - LOF
  - One class SVM
  - Isolation forest
  - PCA

fasttext_seq2seq

- Train and test sequence model
- Try to predict next embedding using LSTM based torch model
- Compute threshold on train dataset (assume 5% logs are anomalies)
- Test on different thresholds and save statistics

## Description of data files

### Dataset labeled by block (e.g. HDFS)

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