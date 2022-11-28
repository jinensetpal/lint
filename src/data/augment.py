#!/usr/bin/env python3

from ..data.generator import get_dataset
from matplotlib import pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf
from glob import glob
from .. import const
import numpy as np
import sys
import os

def difference(dir, tar):
    return len(glob(dir)) - len(glob(tar))

if __name__ == '__main__':
    augment = tf.keras.Sequential([layers.RandomFlip('horizontal_and_vertical'),
                                   layers.RandomRotation(0.25)])
    train, val, test = get_dataset()
    diff = difference(os.path.join(const.BASE_DIR, *const.DATA_PATHS[0], 'leopard', '*'),
                      os.path.join(const.BASE_DIR, *const.DATA_PATHS[0], 'orca', '*'))

    while diff > 0:
        for idx, (X, y) in enumerate(train):
            for img, label in zip(X, y):
                if label == 1:
                    res = augment(img)
                    plt.imsave(os.path.join(const.BASE_DIR, *const.DATA_PATHS[0], 'orca', f'augmented_{diff}.jpg'), res.numpy().astype(np.uint8))

                    diff -= 1
                    if diff == 0:
                        print('[!] Augmentation Complete!')
                        sys.exit(0)
