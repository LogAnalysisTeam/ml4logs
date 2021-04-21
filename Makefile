PROJECT_NAME = ml4logs
export PROJECT_DIR = $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
export ML4LOGS_PYTHON = python

# The default shell is "bash", for cluster environment you would most like want to use a workload manager instead
# e.g., for Slurm , set the ML4LOGS_SHELL variable to "sbatch"
ML4LOGS_SHELL := $(if $(ML4LOGS_SHELL),$(ML4LOGS_SHELL),bash)
$(info Using ML4LOGS_SHELL="$(ML4LOGS_SHELL)")

.PHONY: requirements \
		hdfs1_100k_data hdfs1_100k_preprocess hdfs1_100k_train_test \
		hdfs1_data hdfs1_preprocess hdfs1_train_test \
		bgl_100k thunderbird_100k

hdfs1_100k_data:
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/HDFS1_100k/data.batch"

hdfs1_100k_preprocess:
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/HDFS1_100k/drain_preprocess.batch"
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/HDFS1_100k/fasttext_preprocess.batch"

hdfs1_100k_train_test:
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/HDFS1_100k/drain_loglizer.batch"
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/HDFS1_100k/fasttext_loglizer.batch"
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/HDFS1_100k/fasttext_seq2seq.batch"
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/HDFS1_100k/fasttext_seq2label.batch"

hdfs1_data:
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/HDFS1/data.batch"

hdfs1_preprocess:
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/HDFS1/drain_preprocess.batch"
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/HDFS1/fasttext_preprocess.batch"

hdfs1_train_test:
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/HDFS1/drain_loglizer.batch"
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/HDFS1/fasttext_loglizer.batch"
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/HDFS1/fasttext_seq2seq.batch"
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/HDFS1/fasttext_seq2label.batch"

bgl_100k:
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/bgl_100k.batch"

thunderbird_100k:
	$(ML4LOGS_SHELL) "$(PROJECT_DIR)/scripts/thunderbird_100k.batch"
