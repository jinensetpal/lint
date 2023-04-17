#!/usr/bin/env python3

from tensorflow.keras.models import load_model
from ..data.generator import get_dataset
from ..model.loss import CAMLoss
import tensorflow as tf
from .. import const
import numpy as np
import sys
import os


def group_accuracy(model, gen):
    grp = {}
    for (x, y, p) in gen:
        pred = model(x)
        if type(pred) == list: pred = pred[0]
        acc = tf.experimental.numpy.ravel(tf.cast(pred > 0.5, tf.float32)) == y

        for label in [0, 1]:
            for place in [0, 1]:
                if label not in grp: grp[label] = {}
                if place not in grp[label]: grp[label][place] = np.empty(0)
                grp[label][place] = np.hstack([grp[label][place], acc[np.logical_and(y == label, p == place)].numpy()])

    for label in [0, 1]:
        for place in [0, 1]:
            grp[label][place] = grp[label][place].sum() / grp[label][place].shape[0]
    return grp


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME

    model = load_model(os.path.join(const.BASE_DIR, *const.SAVED_MODEL_PATH, name),
                       custom_objects={'CAMLoss': CAMLoss},
                       compile=False)

    for gen in get_dataset(state='evaluation'): print(gen.split, group_accuracy(model, gen))
