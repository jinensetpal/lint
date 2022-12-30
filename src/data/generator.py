#!/usr/bin/env python3

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import keras
from .. import const
import scipy as sp
import numpy as np
import os


def get_dataset():
    return [image_dataset_from_directory(os.path.join(const.BASE_DIR, *path),
                                         image_size=const.IMAGE_SIZE,
                                         batch_size=const.BATCH_SIZE,
                                         seed=const.SEED) for path in const.DATA_PATHS]


def get_class_activation_map(model, img):
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    label_index = np.argmax(predictions[0])
    class_weights = model.layers[-1].get_weights()[0][:, label_index]

    final_conv_layer = model.get_layer(const.PENULTIMATE_LAYER)

    get_output = keras.backend.function(model.layers[0].input, final_conv_layer.output)
    final_output = extrapolate(*get_output(img), class_weights)

    return final_output, label_index


def extrapolate(conv_outputs, class_weights):
    conv_outputs = np.squeeze(conv_outputs)
    mat_for_mult = sp.ndimage.zoom(conv_outputs, (const.IMAGE_SIZE[0] / conv_outputs.shape[0], const.IMAGE_SIZE[1] / conv_outputs.shape[1], 1), order=1)
    return np.dot(mat_for_mult.reshape((const.IMAGE_SIZE[0] * const.IMAGE_SIZE[1], 32)), class_weights).reshape(const.IMAGE_SIZE[0], const.IMAGE_SIZE[1])


if __name__ == '__main__':
    # imports for visualizing cams
    from tensorflow.keras.models import load_model
    from cv2 import resize, INTER_CUBIC
    from ..model.loss import CAMLoss
    import matplotlib.pyplot as plt
    from pathlib import Path
    from PIL import Image
    import sys

    train, val, test = get_dataset()
    name = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME
    model = load_model(os.path.join(const.BASE_DIR, *const.PROD_MODEL_PATH, name),
                       custom_objects={'CAMLoss': CAMLoss},
                       compile=False)

    fig = plt.figure(figsize=(14, 14),
                     facecolor='white')

    for idx, (X, y) in enumerate(zip(*test.__iter__().next())):
        X = X.numpy()
        if idx == 16: break

        img = resize(X, dsize=const.IMAGE_SIZE, interpolation=INTER_CUBIC)
        out, pred = get_class_activation_map(model, img)
        img = resize(X, dsize=const.IMAGE_SIZE, interpolation=INTER_CUBIC)
        img = Image.fromarray(img.astype('uint8'), 'RGB')

        fig.add_subplot(4, 4, idx + 1)
        plt.imshow(img, alpha=0.5)
        plt.imshow(out, cmap='jet', alpha=0.5)
    plt.tight_layout()

    Path(os.path.join(const.BASE_DIR, *const.CAMS_SAVE_DIR)).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(const.BASE_DIR, *const.CAMS_SAVE_DIR, f'{name}.png'))
