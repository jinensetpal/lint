#!/usr/bin/env python3

from astropy.convolution import Gaussian2DKernel
from tensorflow.keras.losses import Loss
from ..data.generator import extrapolate
import tensorflow as tf
from .. import const


class CAMLoss(Loss):
    def __init__(self, _):
        super().__init__()
        self.weights = None
        self.kernel = tf.convert_to_tensor(Gaussian2DKernel(const.IMAGE_SIZE[0] * .3, x_size=const.IMAGE_SIZE[0], y_size=const.IMAGE_SIZE[1]), dtype=tf.float32)
        self.kernel -= 1E2 * tf.reduce_max(self.kernel)

    def call(self, labels, conv_outputs):
        def compute_loss(conv_output):
            return tf.tensordot(extrapolate(conv_output, self.weights[:, 0], symbolic=True), self.kernel, 1)

        return tf.reduce_mean(tf.map_fn(fn=compute_loss, elems=conv_outputs)) ** 2


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    loss = CAMLoss(const.LOSS_WEIGHTS[0])
    plt.imshow(loss.kernel)
    plt.show()
