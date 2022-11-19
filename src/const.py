#!/usr/bin/env python3

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.as_posix()
DATA_PATHS = [['data', 'train'], ['data', 'test']]

N_CHANNELS = 3
N_CLASSES = 2
N_RES_BLOCKS = 3
N_EPOCHS = 10

TYPE = 'before'
PROD_MODEL_PATH = ['model']

IMAGE_SIZE = (256, 256)
IMAGE_SHAPE = IMAGE_SIZE + (N_CHANNELS,)

MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
