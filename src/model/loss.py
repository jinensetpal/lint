#!/usr/bin/env python3

from tensorflow.keras.losses import Loss, BinaryCrossentropy
from ..data.generator import extrapolate
from tensorflow.math import log
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
        loss = []
        if self.epoch > self.limit: return 0

        for conv_output in conv_outputs:
            weights = self.weights[:, 0]
            cam = extrapolate(conv_output, weights)

            cam -= cam.min()
            cam /= cam.max()

            loss.append(cam[:round(cam.shape[0] * .1), :].mean())
        return log(np.array(loss) + 1)
