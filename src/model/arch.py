#!/usr/bin/env python3

from tensorflow.keras import layers
import tensorflow as tf
from .. import const


class Model(tf.keras.Model):
    def __init__(self, **args):
        super().__init__(**args)
        from tensorflow.python.ops.numpy_ops import np_config
        np_config.enable_numpy_behavior()

    def train_step(self, data):
        weights = self.trainable_variables
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            if const.MODEL_NAME != self.name: self.compiled_loss._losses[1].weights = weights[-2]
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, weights)

        self.optimizer.apply_gradients(zip(gradients, weights))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


def get_model(input_shape, classes, name, channels=3, multiheaded=True):
    input = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(16, kernel_size=(2, 2), padding='same')(input)
    for filters in [16, 64, 32]:
        x = layers.Conv2D(filters, kernel_size=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    relu = layers.ReLU(name='activation_mapping')(x)
    x = layers.Flatten()(relu)
    for units in [128, 128, 64, 32]: x = layers.Dense(units, activation='relu')(x)

    x = layers.Dense(1, activation='sigmoid', name='output')(x)
    outputs = [x, relu] if multiheaded else x

    return Model(inputs=input, outputs=outputs, name=name)
