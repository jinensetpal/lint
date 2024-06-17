#!/usr/bin/bash

#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100_32GB
#SBATCH --time=2:00:00

source ~/.bashrc
source load lint

cd ~/git/lint

MLFLOW_TRACKING_USERNAME=jinensetpal \
MLFLOW_TRACKING_PASSWORD=$MLFLOW_TOKEN \
dvc repro -f
