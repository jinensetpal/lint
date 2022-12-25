#!/usr/bin/env python3

from ..data.generator import get_dataset
from .arch import get_model
from .loss import CAMLoss
import tensorflow as tf
from .. import const
import numpy as np
import mlflow
import sys
import os

def get_callbacks():
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)

    return es

if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME

    train, val, test = get_dataset()
    model = get_model(const.IMAGE_SHAPE, const.N_CLASSES, name, const.N_CHANNELS, multiheaded=const.MODEL_NAME != name)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=const.LEARNING_RATE,
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=1e-08)
    losses = ['binary_crossentropy', CAMLoss()] if const.MODEL_NAME != name else 'binary_crossentropy'
    model.compile(optimizer=optimizer,
                  loss=losses,
                  metrics={'output': 'accuracy'},
                  run_eagerly=True)

    mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    mlflow.tensorflow.autolog()
    with mlflow.start_run():
        model.fit(train,
                  epochs=const.EPOCHS,
                  validation_data=val,
                  use_multiprocessing=True,
                  callbacks=get_callbacks())

        metrics = model.evaluate(test)
        model.save(os.path.join(const.BASE_DIR, *const.PROD_MODEL_PATH, name))
