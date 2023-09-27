#!/usr/bin/env python3

from skimage.transform import resize
from itertools import permutations
from pycocotools.coco import COCO
from .. import const
import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split=None, state='training'):
        coco = COCO(const.ANNOTATIONS_PATH)

        # intentionally lossy mask render
        self.masks = torch.tensor(np.array(list(map(lambda x: resize(resize(resize(coco.annToMask(x), const.IMAGE_SIZE), const.CAM_SIZE), const.IMAGE_SIZE),
                                                    coco.loadAnns(range(len(coco.imgs))))))).to(torch.float).unsqueeze(1).repeat(1, 3, 1, 1)
        self.inv_masks = torch.abs(self.masks - 1)
        self.order = [(*combination, neg) for neg in range(self.masks.shape[0]) for combination in list(permutations(range(self.masks.shape[0]), 2))]

    def __len__(self):
        return len(self.order) * 2

    def __getitem__(self, idx):
        # returns anchor, positive, negative
        if idx > len(self) / 2:
            # inverse masks
            idx = self.order[idx % 2]
            return (self.inv_masks[idx[0]], self.inv_masks[idx[1]], self.masks[idx[2]])
        # regular masks
        idx = self.order[idx % 2]
        return (self.masks[idx[0]], self.masks[idx[1]], self.inv_masks[idx[2]])

    def means(self, model):
        return [model(x.to(const.DEVICE)).mean(axis=0) for x in (self.masks, self.inv_masks)]


if __name__ == '__main__':
    dataset = Dataset()
    print(dataset[0])
