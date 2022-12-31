#!/usr/bin/env python3

from .loss import CAMLoss
import tensorflow as tf
from PIL import Image
from .. import const
import numpy as np
import sys
import os

if __name__ == '__main__':
    if len(sys.argv) < 3: raise ValueError('Enter the model name and image filepaths and command line arguments!')

    model = tf.keras.models.load_model(os.path.join(const.BASE_DIR, *const.PROD_MODEL_PATH, sys.argv[1]),
                                       custom_objects={'CAMLoss': CAMLoss},
                                       compile=False)

    for path in sys.argv[2:]:
        img = tf.expand_dims(tf.image.resize(np.array(Image.open(path))[..., :3], size=const.IMAGE_SIZE), axis=0)

        if model.predict(img)[0][0] > .5: print(f'{path} is an Orca')
        else: print(f'{path} is a Leopard')
