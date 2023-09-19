#!/usr/bin/env python3

from ..data.generator import get_generators
from ..model.loss import CAMLoss
from ..model.arch import Model
from .. import const
import numpy as np
import torch
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
        grp[const.ENCODINGS['label'][label]] = {}
        for place in [0, 1]:
            grp[const.ENCODINGS['label'][label]][const.ENCODINGS['place'][place]] = grp[label][place].sum() / grp[label][place].shape[0]
        del grp[label]
    return grp


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME

    model = Model(input_shape=const.IMAGE_SHAPE).to(const.DEVICE)
    model.load_state_dict(const.MODEL_SAVE_DIR / f'{name}.pt')
    model.eval()

    for gen in get_generators(state='evaluation'): print(gen.split, group_accuracy(model, gen))
