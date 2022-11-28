#!/usr/bin/env python3

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import keras
from .. import const
import scipy as sp
import numpy as np
import os

def get_datasets():
    return [image_dataset_from_directory(os.path.join(const.BASE_DIR, *path),
                                             image_size=const.IMAGE_SIZE,
                                         batch_size=const.BATCH_SIZE) for path in const.DATA_PATHS]

def get_class_activation_map(model, img):
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    label_index = np.argmax(predictions)
    class_weights = model.layers[-1].get_weights()[0][:, label_index] 

    final_conv_layer = model.get_layer(const.PENULTIMATE_LAYER)

    get_output = keras.backend.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    conv_outputs, predictions = get_output([img])
    conv_outputs = np.squeeze(conv_outputs)
    mat_for_mult = sp.ndimage.zoom(conv_outputs, (const.IMAGE_SIZE[0] / conv_outputs.shape[0], const.IMAGE_SIZE[1] / conv_outputs.shape[1], 1), order=1) 
    final_output = np.dot(mat_for_mult.reshape((const.IMAGE_SIZE[0] * const.IMAGE_SIZE[1], 32)), class_weights).reshape(const.IMAGE_SIZE[0], const.IMAGE_SIZE[1]) 

    return final_output, label_index

if __name__ == '__main__':
    # imports for visualizing cams
    from tensorflow.keras.models import load_model
    from cv2 import resize, INTER_CUBIC
    import matplotlib.pyplot as plt
    from pathlib import Path
    from PIL import Image

    train, val, test = get_datasets()
    model = load_model(os.path.join(const.BASE_DIR, *const.PROD_MODEL_PATH, 'default'))

    fig = plt.figure(figsize=(14, 14),
                    facecolor='white')

    for idx, (X, y) in enumerate(zip(*test.__iter__().next())):
        X = X.numpy()

        img = resize(X, dsize=const.IMAGE_SIZE, interpolation=INTER_CUBIC)
        out, pred = get_class_activation_map(model, img)
        img = resize(X, dsize=const.IMAGE_SIZE, interpolation=INTER_CUBIC)
        img = Image.fromarray(img.astype('uint8'), 'RGB')

        fig.add_subplot(4, 4, idx + 1)
        buf = f'Prediction: {pred}, Label: {y}'
        plt.xlabel(buf)
        plt.imshow(img, alpha=0.5)
        plt.imshow(out, cmap='jet', alpha=0.5)
    plt.tight_layout()

    Path(os.path.join(const.BASE_DIR, *const.CAMS_SAVE_DIR)).mkdir(parents=True, exist_ok=True)
    fig.savefig(os.path.join(const.BASE_DIR, *const.CAMS_SAVE_DIR, 'cams.png'))
