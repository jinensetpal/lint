#!/usr/bin/env python3

from tensorflow.keras.activations import relu
from tensorflow.keras.losses import Loss
from ..data.generator import extrapolate
from .. import const


class CAMLoss(Loss):
    def __init__(self):
        super().__init__()
        self.weights = None

    def call(self, labels, conv_outputs):
        loss = 0
        for conv_output in conv_outputs:
            weights = self.weights[:, 0]
            cam = relu(extrapolate(conv_output, weights)).numpy()

            loss += cam[:round(cam.shape[0] * .1), :].sum()
        return loss / conv_outputs.shape[0] * const.SCALE_FACTOR
