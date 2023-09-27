#!/usr/bin/env python3

from pathlib import Path
import torch

# directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
DB_PATH = DATA_DIR / 'db'
CAMS_SAVE_DIR = DATA_DIR / 'cams'
SAVE_MODEL_PATH = BASE_DIR / 'models'
ANNOTATIONS_PATH = DATA_DIR / 'annotations' / 'result.json'

# training
MODEL_NAME = 'default'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1E-3
BATCH_SIZE = 96
MOMENTUM = 0.9
EPOCHS = 50
LOSS_WEIGHTS = [1, 1E-4]  # CSE, CAM

# siamese
S_ALPHA = 20  # for triplet loss
S_EPOCHS = 5
CAM_SIZE = (7, 7)
S_BATCH_SIZE = 16
USE_SIAMESE_LOSS = True

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
LOG_REMOTE = True
