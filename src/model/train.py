#!/usr/bin/env python3

from tensorflow.keras.utils import image_dataset_from_directory
from ..data.generator import get_dataset
from tensorflow.keras import layers
from .layers import ResidualBlock
from .loss import CAMLoss
import tensorflow as tf
from .. import const
import mlflow
import sys
import os

def get_model(input_shape, classes, name, channels=3, multiheaded=True):
    input = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(16, kernel_size=(2,2), padding='same')(input)
    for filters in [16, 64, 32]:
        x = layers.Conv2D(filters, kernel_size=(2,2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    relu = layers.ReLU(name='relu')(x)
    x = layers.Flatten()(relu)
    for units in [128, 128, 64, 32]: x = layers.Dense(units, activation='relu')(x)

    x = layers.Dense(1, activation='softmax', name='output')(x)
    outputs = [x, relu] if multiheaded else x

    return tf.keras.Model(inputs=input, outputs=outputs, name=name)

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
            metrics={'output': 'accuracy'})

    mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    mlflow.tensorflow.autolog()
    # with mlflow.start_run():
    if True:
        model.fit(train,
                  epochs=const.EPOCHS,
                  validation_data=val,
                  use_multiprocessing=True,
                  callbacks=get_callbacks())

        metrics = model.evaluate(test)
        model.save(os.path.join(const.BASE_DIR, *const.PROD_MODEL_PATH, name))
