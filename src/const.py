#!/usr/bin/env python3

from pathlib import Path

# directories
BASE_DIR = Path(__file__).parent.parent.as_posix()
DATA_PATH = ['data',]
SAMPLE_SAVE_DIR = ['data', 'samples']
CAMS_SAVE_DIR = ['data', 'cams']
SAVED_MODEL_PATH = ['models',]

# CAMs
PENULTIMATE_LAYER = 'conv5_block3_out'
THRESHOLD = .97

# training
MODEL_NAME = 'default'
LEARNING_RATE = 1E-4
BATCH_SIZE = 24
SHUFFLE = False
MOMENTUM = 0.9
EPOCHS = 30
LIMIT = round(EPOCHS * .7)  # the point after which the routine switches into fine-tuning
SEED = 1024

# dataset
ENCODINGS = {'place': ['land', 'water'],
             'split': ['train', 'valid', 'test']}
IMAGE_SIZE = (224, 224)
N_CHANNELS = 3
N_CLASSES = 2
IMAGE_SHAPE = IMAGE_SIZE + (N_CHANNELS,)

# tracking
MLFLOW_TRACKING_URI = 'https://dagshub.com/jinensetpal/lint.mlflow'
LOG = True
