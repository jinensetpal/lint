#!/usr/bin/env python3

from tensorflow.keras.activations import relu
from tensorflow.keras.losses import Loss
from ..data.generator import extrapolate
from random import randint, seed
from .. import const


class CAMLoss(Loss):
    def __init__(self):
        super().__init__()
        seed(const.SEED)
        self.weights = None
        self.label_index = None

    def call(self, _, conv_outputs):
        loss = 0
        for conv_output in conv_outputs:
            weights = self.weights[:, 0]
            cam = relu(extrapolate(conv_output, weights)).numpy()

            bounds = {'lower': round(.9 * cam.shape[1]),
                      'upper': cam.shape[1]}
            loss += cam[randint(bounds['lower'], bounds['upper'] - 1):bounds['upper'],
                        randint(bounds['lower'], bounds['upper'] - 1):bounds['upper']].sum()

        return loss / conv_outputs.shape[0] * const.SCALE_FACTOR
