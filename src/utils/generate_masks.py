#!/usr/bin/env python3

import matplotlib.pyplot as plt
from .. import const
import numpy as np
import json

if __name__ == '__main__':
    annotations = json.load(open(const.DATA_DIR / 'annotations.json'))

    for annotation in annotations:
        points = np.array(annotation['annotations'][0]['result'][0]['value']['points'])

        plt.figure(figsize=list(map(lambda x: x / 100, const.IMAGE_SIZE)))
        plt.axis('equal')
        plt.fill(points[:, 0], points[:, 1])
        plt.show()
