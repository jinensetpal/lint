#!/usr/bin/env python3

from ..data.generator import get_dataset
from .loss import CAMLoss, BCELoss
from .arch import get_model
import tensorflow as tf
from .. import const
import mlflow
import sys
import os


def get_callbacks(threshold, parameter='val_loss'):
    def schedule(epoch, lr):
        if epoch != threshold: return lr
        return lr * 1E-2

    plt = tf.keras.callbacks.ReduceLROnPlateau(parameter, patience=const.EPOCHS * .5)  # NOQA F841
    sch = tf.keras.callbacks.LearningRateScheduler(schedule)

    return [sch]


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME
    multiheaded = const.MODEL_NAME != name

    train, val, test = get_dataset()
    model = get_model(const.IMAGE_SHAPE, const.N_CLASSES, name, const.N_CHANNELS, multiheaded=multiheaded)
    model.summary()

    optimizer = tf.keras.optimizers.SGD(learning_rate=const.LEARNING_RATE)
    losses = [BCELoss(const.LIMIT), CAMLoss(const.LIMIT)] if const.MODEL_NAME != name else 'binary_crossentropy'
    model.compile(optimizer=optimizer,
                  loss=losses,
                  metrics={'output': 'accuracy'})

    if const.LOG: mlflow.tensorflow.autolog()
    model.fit(train,
              epochs=const.EPOCHS,
              validation_data=val,
              use_multiprocessing=True,
              callbacks=get_callbacks(const.LIMIT))
    metrics = model.evaluate(test)
    model.save(os.path.join(const.BASE_DIR, *const.PROD_MODEL_PATH, name))
