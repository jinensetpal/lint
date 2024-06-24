#!/usr/bin/env python3

from itertools import pairwise
from src import const
from torch import nn
import torch


class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        units = [3, 16, 64, 32]
        self.convs = nn.ModuleList([nn.Conv2d(*units[:2], 2, padding='same')] +
                                   [nn.Conv2d(*features, 2, 2) for features in pairwise(units[1:])])
        self.convs[-1].register_forward_hook(self._hook)

        self.batchnorms = [nn.BatchNorm2d(n_features, device=const.DEVICE) for n_features in units[1:]]
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.feature_grad = None
        self.feature_rect = None

        self.linear = nn.LazyLinear(2)
        self.softmax = nn.Softmax(dim=1)  # ~equivalent to sigmoid since classes = 2; relevant for CAMs

        self.to(const.DEVICE)
        self(torch.randn(1, *input_shape).to(const.DEVICE))  # initialization
        const.CAM_SIZE = tuple(self.feature_rect.shape[2:])

    def _hook(self, model, i, o):
        def assign(grad):
            self.feature_grad = grad
        self.feature_rect = o
        o.register_hook(assign)

    def forward(self, x):
        for conv, bn in zip(self.convs, self.batchnorms):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)
        x = self.flatten(x)

        logits = self.linear(x)
        return self.softmax(logits), self._compute_hi_res_cam(logits)

    def _compute_hi_res_cam(self, y):
        cams = torch.zeros(*y.shape, *self.feature_rect.shape[2:])
        for img_idx in range(y.shape[0]):
            for class_idx in range(y.shape[1]):
                y[img_idx, class_idx].backward(retain_graph=True, inputs=self.feature_rect)
                cams[img_idx, class_idx] = (self.feature_rect * self.feature_grad).sum(1)[img_idx]
        return cams


if __name__ == '__main__':
    model = Model(input_shape=const.IMAGE_SHAPE)
    model.eval()

    x = torch.rand((3, *const.IMAGE_SHAPE)).to(const.DEVICE)
    y, cam = model(x)
    print(cam.shape)
