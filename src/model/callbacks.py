#!/usr/bin/env python3

from tensorflow.keras import backend as K
import tensorflow as tf
from .. import const


def get_callbacks(threshold, loss_weights, parameter='val_loss'):
    def schedule(epoch, lr):
        if epoch != threshold: return lr
        return lr * 1E-1

    sch = tf.keras.callbacks.LearningRateScheduler(schedule)
    ls = LossManager(const.LIMIT, loss_weights)

    return [sch, ls]


class LossManager(tf.keras.callbacks.Callback):
    def __init__(self, limit, weights):
        self.limit = limit
        self.weights = weights

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.limit and type(self.weights) == list:
            for weight, target in zip(self.weights, [K.variable(1), K.variable(0)]):
                K.set_value(weight, target)
