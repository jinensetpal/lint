#!/usr/bin/env python3

from tensorflow.keras import backend as K
from ..data.generator import get_dataset
from .callbacks import get_callbacks
from .arch import get_model
from .loss import CAMLoss
import tensorflow as tf
from .. import const
import mlflow
import sys
import os


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME
    multiheaded = const.MODEL_NAME != name

    train, val, test = get_dataset()
    model = get_model(const.IMAGE_SHAPE, const.N_CLASSES, name, const.N_CHANNELS, multiheaded=multiheaded)
    model.summary()

    optimizer = tf.keras.optimizers.SGD(learning_rate=const.LEARNING_RATE)
    loss_weights = [K.variable(1), K.variable(0)] if multiheaded else K.variable(1)
    losses = ['binary_crossentropy', CAMLoss(loss_weights)] if const.MODEL_NAME != name else 'binary_crossentropy'
    model.compile(optimizer=optimizer,
                  loss=losses,
                  loss_weights=loss_weights,
                  metrics={'output': 'accuracy'})

    if const.LOG: mlflow.tensorflow.autolog(log_models=False)
    model.fit(train,
              epochs=const.EPOCHS,
              validation_data=val,
              use_multiprocessing=True,
              callbacks=get_callbacks(const.LIMIT, loss_weights))

    model.compile(optimizer=optimizer,
                  loss=losses,
                  metrics={'output': 'accuracy'})  # recompiling since tensorflow does not serialize backend-tampered variables
    metrics = model.evaluate(test)
    model.save(os.path.join(const.BASE_DIR, *const.PROD_MODEL_PATH, name))
