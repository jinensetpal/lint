#!/usr/bin/bash

#SBATCH --account=gpu
#SBATCH --gres=gpu:1

module load cuda/11.2.2 cudnn/cuda-11.2_8.1.1 anaconda
source activate /home/jsetpal/.conda/envs/cent7/5.1.0-py36/lint
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jsetpal/.conda/envs/cent7/5.1.0-py36/lint/lib/python3.10/site-packages/tensorrt

cd ~/scratch/lint

python -m src.model.train $MODEL_NAME
