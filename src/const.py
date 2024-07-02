#!/usr/bin/env python3

from pathlib import Path
import torch

# directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
CAMS_SAVE_DIR = DATA_DIR / 'cams'
SAVE_MODEL_PATH = BASE_DIR / 'models'
ANNOTATIONS_PATH = DATA_DIR / 'annotations' / 'result.json'

# training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1E-3
SELECT_BEST = True
BATCH_SIZE = 64
MOMENTUM = 0.9
EPOCHS = 20
CORRECT_LABEL_SHIFT = True
USE_SIAMESE_LOSS = False
LOSS_WEIGHTS = [1, 1] if USE_SIAMESE_LOSS else [1, 1E+18]  # CSE, CAM respectively
MODEL_NAME = 'multiloss' if LOSS_WEIGHTS[1] else 'default'

# siamese
TRIPLET = False
S_ALPHA = 20  # for triplet loss
S_L1_ALPHA = 5E-1
S_EPOCHS = 100
S_BATCH_SIZE = 128
S_LEARNING_RATE = 2E-3
S_MODEL_NAME = 'clustering'

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
LOG_REMOTE = False
