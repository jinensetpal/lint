#!/usr/bin/env python3

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.as_posix()
DATA_PATHS = [['data', 'train'], ['data', 'val'], ['data', 'test']]

LEARNING_RATE = 1E-2
N_RES_BLOCKS = 3
BATCH_SIZE = 32
N_CHANNELS = 3
N_CLASSES = 2
EPOCHS = 10
SEED = 1024

CAMS_SAVE_DIR = ['data', 'samples']
PROD_MODEL_PATH = ['models',]

IMAGE_SIZE = (192, 192)
IMAGE_SHAPE = IMAGE_SIZE + (N_CHANNELS,)

MLFLOW_TRACKING_URI = 'https://dagshub.com/jinensetpal/tmls22.mlflow'
PENULTIMATE_LAYER = 're_lu'
THRESHOLD = .97
MODEL_NAME = 'default'
