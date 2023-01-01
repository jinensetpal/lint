#!/usr/bin/env python3

from tensorflow.keras.losses import Loss
from ..data.generator import extrapolate
from tensorflow.math import log


class CAMLoss(Loss):
    def __init__(self):
        super().__init__()
        self.weights = None

    def call(self, labels, conv_outputs):
        loss = 0
        for conv_output in conv_outputs:
            weights = self.weights[:, 0]
            cam = extrapolate(conv_output, weights)

            cam -= cam.min()
            cam /= cam.max()

            loss += cam[:round(cam.shape[0] * .1), :].mean()
        return log(loss / conv_outputs.shape[0] + 1)
