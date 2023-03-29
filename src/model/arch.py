#!/usr/bin/env python3

from tensorflow.keras.applications import resnet50
from tensorflow.keras import layers
import tensorflow as tf
from .. import const


class Model(tf.keras.Model):
    def __init__(self, *args, multiheaded=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.multiheaded = multiheaded
        from tensorflow.python.ops.numpy_ops import np_config
        np_config.enable_numpy_behavior()

    def train_step(self, data):
        weights = self.trainable_variables
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            if self.multiheaded: self.compiled_loss._losses[1].weights = weights[-2]
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, weights)

        self.optimizer.apply_gradients(zip(gradients, weights))
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


def get_model(input_shape, classes, name, channels=3, multiheaded=True):
    classes -= 1 if classes == 2 else 0

    backbone = resnet50.ResNet50(weights='imagenet',
                                 include_top=False,
                                 input_shape=(*input_shape, channels))
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dense(classes, activation='sigmoid' if classes == 1 else 'softmax', name='output')(x)
    relu = backbone.get_layer('conv5_block3_out').output
    outputs = [x, relu] if multiheaded else x

    return Model(inputs=backbone.input, outputs=outputs, name=name, multiheaded=multiheaded)


if __name__ == '__main__':
    import sys

    get_model(const.IMAGE_SIZE, const.N_CLASSES, 'default', multiheaded=bool(len(sys.argv) - 1)).summary()
