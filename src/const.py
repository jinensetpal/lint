#!/usr/bin/env python3

from tensorflow.keras import backend as K
from pathlib import Path

# directories
BASE_DIR = Path(__file__).parent.parent.as_posix()
DATA_PATH = ['data',]
SAMPLE_SAVE_DIR = ['data', 'samples']
CAMS_SAVE_DIR = ['data', 'cams']
SAVED_MODEL_PATH = ['models',]

# CAMs
PENULTIMATE_LAYER = 'conv5_block3_out'

# training
MODEL_NAME = 'default'
BATCH_SIZE = 16
SHUFFLE = True
MOMENTUM = 0.9
EPOCHS = 3
STRATIFIED = True
LEARNING_RATE = [1E-3, 1E-4, 1E-6]
LOSS_WEIGHTS = [[K.variable(7E-1), K.variable(1)],
                [K.variable(8E-1), K.variable(5E2)],
                [K.variable(1), K.variable(0)]]
LIMIT = [1, 2]  # [round(EPOCHS * .2), round(EPOCHS * .8)]  # the point after which the routine switches bootstrap -> training -> fine-tuning
SEED = 1024

# dataset
ENCODINGS = {'place': ['land', 'water'],
             'label': ['landbird', 'waterbird'],
             'split': ['train', 'valid', 'test']}
IMAGE_SIZE = (224, 224)
N_CHANNELS = 3
N_CLASSES = 2
IMAGE_SHAPE = IMAGE_SIZE + (N_CHANNELS,)

# tracking
MLFLOW_TRACKING_URI = 'https://dagshub.com/jinensetpal/lint.mlflow'
LOG = True
