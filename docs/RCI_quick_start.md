# RCI Quick Start
This document gives a quick setup overview of `ml4logs` development environment on [RCI](http://rci.cvut.cz/) cluster.

## Create a virtual environment

Start a short interactive GPU session:
```bash
srun -p gpufast --gres=gpu:1 --time=0:30:00 --pty bash -i
```

Load Python module:
```bash
ml PyTorch/1.7.1-fosscuda-2020b
```

Create the virtual environment:
```bash
python -m venv ml4logs_env
```

Acvtivate it:
```bash
source ml4logs_env/bin/activate
```

You might want to setup Jupyter kernel for the virtual environment:
```bash
pip install jupyter

ipython kernel install --name "ml4logs_env" --user
```

Clone the repository:
```bash
git clone git@github.com:LogAnalysisTeam/ml4logs.git

cd ml4logs
```

Setup for development:
```bash
python setup.py develop
```

Note that during this step all dependencies are installed. You might want to check the log wheter everything went well.

Create `init_environment.sh`:

```bash
echo "initializing environment..."
ml PyTorch/1.7.1-fosscuda-2020b
source {PUT_YOUR_PYTHON_VIRTUALENV_DIRECTORY_HERE}/ml4logs_env/bin/activate

if [[ -z "${ML4LOGS_PYTHON}" ]]; then
    export ML4LOGS_PYTHON=python
fi

echo "ML4LOGS_PYTHON: \"${ML4LOGS_PYTHON}\""
echo "PROJECT_DIR: \"${PROJECT_DIR}\""
echo "done"
```

This file will be automatically `source`d by all run `scripts/`.

## SLURM
All batch files in `scripts/` can be run both locally or on the cluster. RCI uses SLURM where you use
```bash
sbatch scripts/SCRIPT_NAME.batch
```
command to schedule a job. SLURM job configuration is done via [commented lines](https://login.rci.cvut.cz/wiki/jobs) in head of each batch file, so these get ignored when run locally, e.g.:
```bash
bash scripts/SCRIPT_NAME.batch
```
If using the `Makefile` to run the jobs on cluster do not forget to set
```bash
export ML4LOGS_SHELL=sbatch
```

If not set, the `Makefile` defaults to running localy (`bash`).

## Download the Data
Try to run everything on the reduced dataset initially:

```bash
make hdfs1_100k_data
```

## Preprocess the Data
```
make hdfs1_100k_preprocess
```

## Run all HDFS Benchmarks
```
make hdfs1_100k_train_test
```

## Experiments with the Full Dataset
```bash
make hdfs1_data
make hdfs1_preprocess
make hdfs1_train_test
```