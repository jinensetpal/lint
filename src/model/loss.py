#!/usr/bin/env python

class MeanSquaredError(tf.keras.losses.Loss):

  def call(self, y_true, y_pred):
    return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)
