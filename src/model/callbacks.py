#!/usr/bin/env python3

from tensorflow.keras import backend as K
import tensorflow as tf
from .. import const


def get_callbacks(threshold, parameter='val_loss'):
    def schedule(epoch, lr):
        if epoch != threshold: return lr
        return lr * 1E-1

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          patience=25,
                                          restore_best_weights=True,
                                          start_from_epoch=10)
    sch = tf.keras.callbacks.LearningRateScheduler(schedule)
    lm = LossManager(const.LIMIT, const.LOSS_WEIGHTS)

    return [es, sch]


class LossManager(tf.keras.callbacks.Callback):
    def __init__(self, limit, weights):
        self.limit_iter = iter(limit)
        self.target_iter = iter(weights)

        self.targets = next(self.target_iter)
        self.limit = next(self.limit_iter)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in [self.limit, self.limit // 2] and type(self.weights) == list:
            for weight, target in zip(self.weights, self.targets): K.set_value(weight, target)

            self.limit = next(self.limit_iter)
            self.targets = next(self.target_iter)
