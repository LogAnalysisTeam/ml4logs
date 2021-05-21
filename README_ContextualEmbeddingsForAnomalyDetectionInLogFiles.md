ml4logs - Contextual Embeddings for Anomaly Detection in Log Files
==============================
This folder contains the project for using deep neural network models, specifically BERT, to create vector embeddings of log lines for the purposes of anomaly detection
As this project is a fork of the CTU FEE LogAnomalyDetection team repository, only several subfolders pertain specifically to the work done for this thesis.

The main code for the work is in subfolder src/bert, which only original code done for this thesis.
Supplementary code for anomaly detection evaluation of the embeddings, and for general data preprocessing and splitting, some of which was done in cooperation with Koryťák
is present is src/anomaly_detecion and src/data subfolders

The project was run on the RCI cluster using SLURM job scheduler.
The order of the runner SLURM scripts in subfolder scripts the tree structure below shows the basic steps to run this thesis in order.

Project Organization
------------

    ├── LICENSE
    ├── README_ContextualEmbeddingsForAnomalyDetectionInLogFiles.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   └── anomaly_detection     <- Subfolder for anomaly detection experiment models
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`.
    │
    ├── scripts            <- Scripts running the code to reproduce published results The scripts are meant for the SLURM batching system.
    │   ├── tokenize_and_chunkify_dataset.batch <- Script for running src/bert/tokenize_and_chunkify.py
    │   ├── pipeline_tokenize_chunkify_split_combine.batch <- Script for running src/bert/tokenize_chunkify_train-val-split_combine.py
    │   ├── prepare_ict_from_chunks.batch <- Script for running src/bert/prepare_ict_dataset_from_chunks.py
    │   ├── ict_final_drop_columns_flatten_preprocessing.batch <- Script for running src/bert/ict_drop_columns_for_final_preprocessing_step.py
    │   ├── ict_pretrain_preprocessed_env_script.batch <- Script for running src/bert/train_ict_preprocessed_data.py 
    │   ├── embed_labeled_dataset_env_script.batch <- Script for running src/bert/embed_labeled_dataset_for_anomaly_detection.py
    │   ├── aetcn_experiment_env.batch <- Script for running src/anomaly_detection/aetcn_experiment.py
    │   ├── Other SLURM scripts for less important tasks
    │   └── Python helper scripts for mass submission of jobs
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module.
    │   │
    |   ├── anomaly_detection  <- Scripts for anomaly detection experiments specifically for this thesis
    |   |   ├── aetcn_experiment.py - our runner script for anomaly detection experiment using embeddings created by embed_labeled_dataset_for_anomaly_detection.py
    │   │   └── Remaining files - implementation of AETCN anomaly detection method, provided by Koryťák
    |   |
    │   ├── data           <- Scripts to download or generate data.
    │   │   ├── download_dataset.py - Used for downloading datasets
    │   │   └── prepare_hdfs_split.py - Script done with cooperation with Koryťák for correct train-val-test split of HDFS1 dataset
    │   │
    │   ├── bert         <- Main folder for code of this thesis, containing all data preprocessing and training scripts to create sentence encoders for contextual embedding
    │   │   ├── dataset_pipeline.py - Contains helper code for dataset processing pipelines
    │   │   ├── dataset_utils.py - Contains lower level dataset preprocessing functions
    │   │   ├── embed_dataset_using_pretrained_model.py - Runnable script for creating vector embeddings of log dataset
    │   │   ├── embed_labeled_dataset_for_anomaly_detection.py - Runnable script for creating vector embeddings of labeled log datasets
    │   │   ├── encoders.py - Source code for the Sentence Encoder implementations
    │   │   ├── ict_drop_columns_for_final_preprocessing_step.py - Runnable helper script for dropping of unnecessary columns from dataset prior to ICT training
    │   │   ├── ict.py - Source code for ICT pretraining models
    │   │   ├── prepare_ict_dataset_from_chunks.py - Runnable script for creation of ICT training data from already preprocessed chunked dataset
    │   │   ├── tokenize_and_chunkify.py - Runnable script for initial creation of chunked dataset from raw .log text files
    │   │   └── tokenize_chunkify_train-val-split_combine.py - Runnable script for initial creation of more complex chunked dataset by mixing several source raw .log datasets
    │   │   └── train_ict_preprocessed_data.py - Runnable script for main Sentence Encoder training using ICT, accepting datasets created by prepare_ict_dataset_from_chunks



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
