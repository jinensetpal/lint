#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.keras.backend as K

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 filters: int, 
                 dilation_rate: tuple[int, int] = (1,1), 
                 kernel_initializer: str = 'glorot_uniform', 
                 momentum: float = 0.99, 
                 epsilon: float = 0.001, 
                 downsample: bool = False,
                 use_bias: bool = False, 
                 use_sync: bool = True, 
                 kernel_regularizer = None, 
                 bias_regularizer = None):
        
        # parameters Conv2D
        self._filters = filters
        self._dilation_rate = dilation_rate
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._use_bias = use_bias
        self._use_sync = use_sync

        # parameters Batch Norm
        if K.image_data_format() == "channels_last":
            # channels_last: (batch_size, height, width, channels)
            self._axis = -1
        else:
            # not channels_last: (batch_size, channels, height, width)
            self._axis = 1

        self._momentum = momentum 
        self._epsilon = epsilon 

        # downsample
        self._downsample = downsample

        if downsample:
            self._strides = (2,2)
        else:
            self._strides = (1,1)

        self._activation = tf.keras.layers.ReLU()
        super(ResidualBlock, self).__init__()

    def build(self, ishape):
        inputs = tf.keras.Input(shape=ishape[1:])
        x = tf.keras.layers.Conv2D(filters=self._filters, kernel_size=(3, 3), strides=self._strides, padding='same', 
                                   dilation_rate=self._dilation_rate, kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer,
                                   bias_regularizer=self._bias_regularizer, use_bias=self._use_bias)(inputs)
        x = tf.keras.layers.BatchNormalization(axis=self._axis, momentum=self._momentum)(x)
        x = self._activation(x)
        x = tf.keras.layers.Conv2D(filters=self._filters, kernel_size=(3, 3), strides=self._strides, padding='same', 
                                   dilation_rate=self._dilation_rate, kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer,
                                   bias_regularizer=self._bias_regularizer, use_bias=self._use_bias)(x)
        x = tf.keras.layers.BatchNormalization(axis=self._axis, momentum=self._momentum)(x)

        if self._downsample:
          self._strides = list(map(lambda x: x * 2, self._strides))
        res = tf.keras.layers.Conv2D(filters=self._filters, kernel_size=(1, 1), padding='same', strides=self._strides,
                                   dilation_rate=self._dilation_rate, kernel_initializer=self._kernel_initializer, kernel_regularizer=self._kernel_regularizer,
                                   bias_regularizer=self._bias_regularizer, use_bias=self._use_bias)(inputs)
        output = tf.keras.layers.Add()([res, x])
        self.model = tf.keras.Model(inputs, output, name='residual')

        super(ResidualBlock, self).build(ishape)

    def call(self, inputs):
        x = self.model(inputs)
        return self._activation(x)
