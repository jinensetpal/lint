#!/usr/bin/env python3

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.as_posix()
DATA_PATHS = [['data', 'train'], ['data', 'val'], ['data', 'test']]

LEARNING_RATE = 1E-4
N_RES_BLOCKS = 3
BATCH_SIZE = 32
N_CHANNELS = 3
N_CLASSES = 2
EPOCHS = 10

CAMS_SAVE_DIR = ['data', 'samples', 'cams']
PROD_MODEL_PATH = ['models',]

IMAGE_SIZE = (192, 192)
IMAGE_SHAPE = IMAGE_SIZE + (N_CHANNELS,)

MLFLOW_TRACKING_URI = 'https://dagshub.com/jinensetpal/tmls22.mlflow'
PENULTIMATE_LAYER = 'residual_block_2'
