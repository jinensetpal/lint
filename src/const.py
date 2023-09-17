#!/usr/bin/env python3

# from tensorflow.keras import backend as K
from pathlib import Path

# directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
SAMPLE_SAVE_DIR = DATA_DIR / 'samples'
CAMS_SAVE_DIR = DATA_DIR / 'cams'
SAVED_MODEL_PATH = BASE_DIR / 'models'

# hyperparameters
SHOW_CAMS = True
BATCH_SIZE = 16
MOMENTUM = 0.9
EPOCHS = 3
LEARNING_RATE = [1E-3, 1E-4, 1E-6]
# LOSS_WEIGHTS = [[K.variable(7E-1), K.variable(1)],
#                 [K.variable(8E-1), K.variable(5E2)],
#                 [K.variable(1), K.variable(0)]]
LIMIT = [1, 2]  # [round(EPOCHS * .2), round(EPOCHS * .8)]  # the point after which the routine switches bootstrap -> training -> fine-tuning
SEED = 1024

# dataset
ENCODINGS = {'place': ['land', 'water'],
             'label': ['landbird', 'waterbird'],
             'split': ['train', 'valid', 'test']}
IMAGE_SIZE = (224, 224)
N_CHANNELS = 3
N_CLASSES = 2
IMAGE_SHAPE = (N_CHANNELS,) + IMAGE_SIZE

# logging
MLFLOW_TRACKING_URI = 'https://dagshub.com/jinensetpal/lint.mlflow'
LOG = True
