#!/usr/bin/env python3

from ..data.generator import get_dataset
from .arch import get_model
from .loss import CAMLoss
import tensorflow as tf
from .. import const
import mlflow
import sys
import os


def get_callbacks(parameter):
    lr = tf.keras.callbacks.ReduceLROnPlateau(parameter, patience=const.EPOCHS * .25)

    return [lr,]


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME
    multiheaded = const.MODEL_NAME != name

    train, val, test = get_dataset()
    model = get_model(const.IMAGE_SHAPE, const.N_CLASSES, name, const.N_CHANNELS, multiheaded=multiheaded)
    model.summary()

    optimizer = tf.keras.optimizers.SGD(learning_rate=const.LEARNING_RATE)
    losses = ['binary_crossentropy', CAMLoss()] if const.MODEL_NAME != name else 'binary_crossentropy'
    model.compile(optimizer=optimizer,
                  loss=losses,
                  metrics={'output': 'accuracy'},
                  run_eagerly=multiheaded)

    if const.LOG: mlflow.tensorflow.autolog()
    model.fit(train,
              epochs=const.EPOCHS,
              validation_data=val,
              use_multiprocessing=True,
              callbacks=get_callbacks('relu_loss') if multiheaded else get_callbacks('val_loss'))
    metrics = model.evaluate(test)
    model.save(os.path.join(const.BASE_DIR, *const.PROD_MODEL_PATH, name))
