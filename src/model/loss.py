#!/usr/bin/env python3

from tensorflow.keras.activations import sigmoid
from tensorflow.keras.losses import Loss
from ..data.generator import extrapolate
import tensorflow as tf


class CAMLoss(Loss):
    def __init__(self, _):
        super().__init__()
        self.weights = None

    def call(self, labels, conv_outputs):
        def compute_loss(conv_output):
            weights = self.weights[:, 0]
            cam = extrapolate(conv_output, weights)

            cam -= cam.min()
            cam /= cam.max()
            cam = sigmoid(cam)

            return cam[:round(cam.shape[0] * .1), :].mean()

        return tf.math.log(tf.map_fn(fn=compute_loss, elems=conv_outputs) + 1)
