#!/usr/bin/env python3

from torchvision.models.resnet import resnet50
from itertools import pairwise
from src import const
from torch import nn
import torch


class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.feature_grad = None
        self.feature_vect = None
        self.backbone = resnet50()
        self.backbone.fc = nn.LazyLinear(2)
        self.softmax = nn.Softmax(dim=1)  # ~equivalent to sigmoid since classes = 2; for CAMs
        self.backbone.layer4[-1].conv3.register_forward_hook(self.hook)

    def hook(self, model, i, o):
        def assign(grad):
            self.feature_grad = grad
        self.feature_vect = o
        o.register_hook(assign)

    def forward(self, x):
        x = self.backbone(x)
        x = self.softmax(x)

        return x, self._compute_hrs_cam(x)

    def _compute_hrs_cam(self, y):
        cams = torch.zeros(*y.shape, 7, 7)
        for img_idx in range(y.shape[0]):
            for class_idx in range(y.shape[1]):
                (y[img_idx, class_idx]).backward(retain_graph=True)
                cams[img_idx, class_idx] = (self.feature_vect * self.feature_grad).sum(1)[img_idx]
        return cams


if __name__ == '__main__':
    model = Model(input_shape=const.IMAGE_SHAPE).to(const.DEVICE)
    model.eval()
    print(model)

    x = torch.rand((3, *const.IMAGE_SHAPE)).to(const.DEVICE)
    y, hrs_cam = model(x)
    print(hrs_cam.shape)
