#!/usr/bin/env python3

from pathlib import Path

# directories
BASE_DIR = Path(__file__).parent.parent.as_posix()
DATA_PATHS = [['data', 'train'], ['data', 'val'], ['data', 'test']]
CAMS_SAVE_DIR = ['data', 'cams']
PROD_MODEL_PATH = ['models',]

# CAMs 
PENULTIMATE_LAYER = 'relu'
THRESHOLD = .97

# training 
MODEL_NAME = 'default'
LEARNING_RATE = 1E-2
SCALE_FACTOR = 5
BATCH_SIZE = 32
EPOCHS = 10
SEED = 1024

# dataset
IMAGE_SIZE = (192, 192)
N_CHANNELS = 3
N_CLASSES = 2
IMAGE_SHAPE = IMAGE_SIZE + (N_CHANNELS,)

# tracking
MLFLOW_TRACKING_URI = 'https://dagshub.com/jinensetpal/tmls22.mlflow'
