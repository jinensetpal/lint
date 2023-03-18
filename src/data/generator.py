#!/usr/bin/env python3

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path, PosixPath
from tensorflow import keras
import tensorflow as tf
from PIL import Image
from .. import const
import pandas as pd
import numpy as np
import random
import os


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, batch_size=32, dim=(224, 224), n_channels=1,
                 shuffle=True, split='train', seed=0):
        'Initialization'
        if type(df) == PosixPath: df = pd.read_csv(df) 
        self.df = df
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        if split not in const.ENCODINGS['split']: split = 'all'
        self.split = split
        self.indexes = {}
        self.on_epoch_end()
        random.seed(seed)
        self.gen = ImageDataGenerator()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.df.shape[0] / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[self.split][index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        ids = [self.df.iloc[k] for k in indexes]

        # Generate data
        return self.__data_generation(ids)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        for split in range(3):
            self.indexes[const.ENCODINGS['split'][split]] = np.array(self.df[self.df['split'] == split]['img_id'])
        self.indexes['all'] = np.arange(self.df.shape[0]) 

        if self.shuffle: 
            for split in self.indexes: np.random.shuffle(self.indexes[split])

    def __data_generation(self, IDs):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=object)
        
        # Generate data
        for idx, id in enumerate(IDs):
            X[idx] = Image.open(os.path.join(const.BASE_DIR, *const.DATA_PATH, id['img_filename'])).resize(const.IMAGE_SIZE)
            y[idx] = id['y']

        return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32)



def get_dataset():
    return [DataGenerator(Path(os.path.join(const.BASE_DIR, *const.DATA_PATH, 'metadata.csv')), 
                          const.BATCH_SIZE, const.IMAGE_SIZE, const.N_CHANNELS, 
                          split=split) for split in const.ENCODINGS['split']]


def get_class_activation_map(model, img):
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    label_index = np.argmax(predictions[0])
    class_weights = model.layers[-1].get_weights()[0][:, label_index]

    final_conv_layer = model.get_layer(const.PENULTIMATE_LAYER)

    get_output = keras.backend.function(model.layers[0].input, final_conv_layer.output)
    final_output = extrapolate(*get_output(img), class_weights)

    return final_output, label_index


def extrapolate(conv_outputs, class_weights, symbolic=False):
    conv_outputs = tf.squeeze(conv_outputs)
    mat_for_mult = tf.image.resize(conv_outputs, const.IMAGE_SIZE)

    if symbolic: return tf.tensordot(mat_for_mult.reshape((const.IMAGE_SIZE[0] * const.IMAGE_SIZE[1], 32)), class_weights, 1).reshape(const.IMAGE_SIZE[0], const.IMAGE_SIZE[1])
    return np.dot(mat_for_mult.numpy().reshape((const.IMAGE_SIZE[0] * const.IMAGE_SIZE[1], 32)), class_weights).reshape(const.IMAGE_SIZE[0], const.IMAGE_SIZE[1])


if __name__ == '__main__':
    # imports for visualizing cams
    from tensorflow.keras.models import load_model
    from cv2 import resize, INTER_CUBIC
    from ..model.loss import CAMLoss
    import matplotlib.pyplot as plt
    from PIL import Image
    import sys

    train, val, test = get_dataset()
    name = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME
    model = load_model(os.path.join(const.BASE_DIR, *const.PROD_MODEL_PATH, name),
                       custom_objects={'CAMLoss': CAMLoss},
                       compile=False)

    fig = plt.figure(figsize=(14, 14),
                     facecolor='white')

    Path(os.path.join(const.BASE_DIR, *const.CAMS_SAVE_DIR, name)).mkdir(parents=True, exist_ok=True)
    for idx, (X, y) in enumerate(zip(*test.__iter__().next())):
        X = X.numpy()
        if idx == 16: break

        img = resize(X, dsize=const.IMAGE_SIZE, interpolation=INTER_CUBIC)
        out, pred = get_class_activation_map(model, img)
        img = resize(X, dsize=const.IMAGE_SIZE, interpolation=INTER_CUBIC)
        img = Image.fromarray(img.astype('uint8'), 'RGB')

        plt.figure(1)
        fig.add_subplot(4, 4, idx + 1)
        plt.imshow(img, alpha=0.5)
        plt.imshow(out, cmap='jet', alpha=0.5)

        plt.figure(2)
        plt.imshow(img, alpha=0.5)
        plt.imshow(out, cmap='jet', alpha=0.5)
        plt.savefig(os.path.join(const.BASE_DIR, *const.CAMS_SAVE_DIR, name, f'{idx}.png'))
        plt.clf()
    plt.tight_layout()

    Path(os.path.join(const.BASE_DIR, *const.CAMS_SAVE_DIR)).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(const.BASE_DIR, *const.CAMS_SAVE_DIR, f'{name}.png'))
