#!/usr/bin/env python3

from tensorflow.keras.utils import image_dataset_from_directory
from .. import const
import os

def get_datasets():
    return [image_dataset_from_directory(os.path.join(const.BASE_DIR, *path),
                                         image_size=const.IMAGE_SIZE,
                                         batch_size=const.BATCH) for path in const.DATA_PATHS]

if __name__ == '__main__':
    train, val, test = get_datasets()
    print(train.take(1))
    print(val.take(1))
    print(test.take(1))
