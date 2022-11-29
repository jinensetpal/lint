#!/usr/bin/env python

from tensorflow.keras.losses import Loss
from random import randint
import tensorflow as tf
from .. import const

class CAMLoss(Loss):
    def call(self, _, slice):
        bounds = {'lower': round(.8 * slice.shape[1]), 
                  'upper': slice.shape[1]}
        return tf.reduce_mean(slice[0, 
                                    randint(bounds['lower'], bounds['upper'] - 1):bounds['upper'], 
                                    randint(bounds['lower'], bounds['upper'] - 1):bounds['upper'], 
                                    :], axis=-1) * const.SCALE_FACTOR 
