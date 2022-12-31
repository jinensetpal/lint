#!/usr/bin/env python3

from tensorflow.keras.activations import relu
from tensorflow.keras.losses import Loss
from ..data.generator import extrapolate
from random import randint, seed
import matplotlib.pyplot as plt
from matplotlib import patches
from .. import const
import os


class CAMLoss(Loss):
    def __init__(self):
        super().__init__()
        seed(const.SEED)
        self.weights = None
        self.idx = 0

    def call(self, labels, conv_outputs):
        loss = 0
        for conv_output in conv_outputs:
            weights = self.weights[:, 0]
            cam = relu(extrapolate(conv_output, weights)).numpy()

            bounds = {'lower': round(.1 * cam.shape[1]),
                      'upper': 0}

            lower = randint(0, round(cam.shape[1] * .8))
            upper = randint(lower + 10, cam.shape[1])
            loss += cam[:round(cam.shape[0] * .1),
                        lower:upper].sum()

            # DEBUG
            plt.imshow(cam, cmap='jet')
            plt.gca().add_patch(patches.Rectangle(
                (0, 0), 185, 25,
                linewidth=1, 
                edgecolor='r', 
                facecolor='none'))
            plt.savefig(os.path.join(const.BASE_DIR, 'debug', f'{self.idx}.png'))
            plt.clf()
            plt.imshow(cam[:25, lower:upper], cmap='jet') # height, width
            plt.savefig(os.path.join(const.BASE_DIR, 'debug', f'{self.idx}-sliced.png'))
            self.idx += 1
            plt.clf()


        return loss / conv_outputs.shape[0] * const.SCALE_FACTOR
