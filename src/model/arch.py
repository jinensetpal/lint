#!/usr/bin/env python3

from itertools import pairwise
from src import const
from torch import nn
import torch


class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.convs = nn.ModuleList([nn.Conv2d(*filters, 2, padding='same') for filters in pairwise([3, 16, 64, 32])])
        self.batchnorms = nn.ModuleList([nn.BatchNorm2d(channels) for channels in [16, 64, 32]])
        self.avgpool = nn.AvgPool2d(2)

        self.penultimate = None
        self.flatten = nn.Flatten()

        self.linears = nn.ModuleList([nn.LazyLinear(units) for units in [128, 128, 64, 32, 2]])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # ~equivalent to sigmoid since classes = 2; for CAMs

    def forward(self, x):
        for conv, bn in zip(self.convs, self.batchnorms):
            x = conv(x)
            self.penultimate = x
            x = bn(x)
            x = self.avgpool(x)
        x = self.flatten(x)

        for linear in self.linears[:-1]:
            x = linear(x)
            x = self.relu(x)

        x = self.linears[-1](x)
        x = self.softmax(x)

        return x, self._compute_cam(torch.argmax(x, dim=1))

    def _compute_cam(self, label_index):
        class_weights = self.linears[-1].weight
        def single_cam(penultimate):
            return torch.vmap(lambda x: torch.tensordot(x, penultimate, dims=1), in_dims=0)(class_weights)

        return torch.vmap(single_cam, in_dims=0)(self.penultimate)


if __name__ == '__main__':
    model = Model(input_shape=const.IMAGE_SHAPE).to(const.DEVICE)
    model.eval()
    print(model)

    x = torch.rand((3, *const.IMAGE_SHAPE)).to(const.DEVICE)
    y, cam = model(x)
    print(cam.shape)
