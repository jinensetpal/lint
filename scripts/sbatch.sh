#!/usr/bin/bash

#SBATCH --account=gpu
#SBATCH --gres=gpu:1
##SBATCH --constraint=V100_32GB
#SBATCH --time=2:00:00

module load cuda cudnn anaconda
source activate lint

cd ~/git/lint

MLFLOW_TRACKING_URI=https://dagshub.com/jinensetpal/lint.mlflow \
MLFLOW_TRACKING_USERNAME=jinensetpal \
MLFLOW_TRACKING_PASSWORD=$MLFLOW_TOKEN \
python -m src.classification.train $MODEL_NAME
