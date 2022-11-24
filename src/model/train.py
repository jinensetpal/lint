#!/usr/bin/env python3

from tensorflow.keras.utils import image_dataset_from_directory
from ..data.generator import get_datasets
from tensorflow.keras import layers
from .layers import ResidualBlock
import tensorflow as tf
from .. import const
import mlflow
import os

def get_model(dim, classes, channels=3):
    model = tf.keras.models.Sequential()	
    model.add(tf.keras.Input(shape=(const.IMAGE_SHAPE)))
    for _ in range(const.N_RES_BLOCKS):
        model.add(ResidualBlock(64, downsample=True))

    model.add(layers.Flatten())
    for units in [128, 64, 32, 16]: model.add(layers.Dense(units, activation='relu'))

    model.add(layers.Dense(1, activation='softmax'))
    return model

def get_callbacks():
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)

    return es


if __name__ == '__main__':
    train, val, test = get_datasets()
    model = get_model(const.IMAGE_SIZE, const.N_CLASSES, const.N_CHANNELS)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1E-4,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-08)


    model.compile(optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy'])

    mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    mlflow.tensorflow.autolog()
    with mlflow.start_run():
        history = model.fit(train,
                            epochs=const.N_EPOCHS,
                            validation_data=val,
                            use_multiprocessing=True,
                            callbacks=get_callbacks())

        trained_model_loss, trained_model_accuracy = model.evaluate(test)
        model.save(os.path.join(const.BASE_DIR, *const.PROD_MODEL_PATH))
