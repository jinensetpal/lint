#!/usr/bin/env python3

from itertools import pairwise
from src import const
from torch import nn
import torch


class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.feature_grad = None
        self.feature_rect = None

        self.convs = nn.ModuleList([nn.Conv2d(*filters, 2, padding='same') for filters in pairwise([3, 16, 64, 32])])
        self.batchnorms = nn.ModuleList([nn.BatchNorm2d(channels) for channels in [16, 64, 32]])
        self.avgpool = nn.AvgPool2d(16)
        self.convs[-1].register_forward_hook(self.hook)

        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # ~equivalent to sigmoid since classes = 2; for CAMs

    def hook(self, model, i, o):
        def assign(grad):
            self.feature_grad = grad
        self.feature_rect = o
        o.register_hook(assign)

    def forward(self, x):
        for conv, bn in zip(self.convs, self.batchnorms):
            x = conv(x)
            x = bn(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        x = self.linear(x)
        x = self.relu(x)
        x = self.softmax(x)

        return x, self._compute_hrs_cam(x)

    def _compute_hrs_cam(self, y):
        cams = torch.zeros(*y.shape, *self.feature_rect.shape[2:])
        for img_idx in range(y.shape[0]):
            for class_idx in range(y.shape[1]):
                (y[img_idx, class_idx]).backward(retain_graph=True)
                cams[img_idx, class_idx] = (self.feature_rect * self.feature_grad).sum(1)[img_idx]
        return cams


if __name__ == '__main__':
    model = Model(input_shape=const.IMAGE_SHAPE).to(const.DEVICE)
    model.eval()
    print(model)

    x = torch.rand((3, *const.IMAGE_SHAPE)).to(const.DEVICE)
    y, hrs_cam = model(x)
    print(hrs_cam.shape)
