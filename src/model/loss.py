#!/usr/bin/env python3

from tensorflow.keras.losses import Loss, BinaryCrossentropy
from ..data.generator import extrapolate
import tensorflow as tf
from .. import const
import numpy as np


class BCELoss(BinaryCrossentropy):
    def __init__(self, limit):
        super().__init__()
        self.epoch = 0
        self.limit = limit

    def call(self, *args, **kwargs):
        factor = 1 if self.epoch > self.limit else 0
        return super().call(*args, **kwargs) * factor


class CAMLoss(Loss):
    def __init__(self, limit):
        super().__init__()
        self.epoch = 0
        self.limit = limit
        self.weights = None

    def call(self, labels, conv_outputs):
        def compute_loss(conv_output):
            weights = self.weights[:, 0]
            cam = extrapolate(conv_output, weights)

            cam -= cam.min()
            cam /= cam.max()

            return cam[:round(cam.shape[0] * .1), :].mean()

        return 0 if self.epoch > self.limit else tf.math.log(tf.map_fn(fn=compute_loss, elems=conv_outputs) + 1)
