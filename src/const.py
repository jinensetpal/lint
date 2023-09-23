#!/usr/bin/env python3

from pathlib import Path
import torch

# directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
SAMPLE_SAVE_DIR = DATA_DIR / 'samples'
CAMS_SAVE_DIR = DATA_DIR / 'cams'
SAVE_MODEL_PATH = BASE_DIR / 'models'

# training
MODEL_NAME = 'default'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1E-3
BATCH_SIZE = 16
MOMENTUM = 0.9
EPOCHS = 50
LOSS_WEIGHTS = [1, 1E6]  # CSE, CAM

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
