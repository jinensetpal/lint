#!/usr/bin/env python3

from src import const
from torch import nn
import torchvision
import torch


class Model(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.backbone = torchvision.models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()

        self.feature_grad = None
        self.feature_rect = None

        self.backbone.layer4[-1].conv3.register_forward_hook(self._hook)

        self.linear = nn.LazyLinear(2)
        self.softmax = nn.Softmax(dim=1)  # ~equivalent to sigmoid since classes = 2; relevant for CAMs

    def _hook(self, model, i, o):
        def assign(grad):
            self.feature_grad = grad
        self.feature_rect = o
        o.register_hook(assign)

    def forward(self, x):
        x = self.backbone(x)
        x = self.linear(x)
        x = self.softmax(x)

        return x, self._compute_hi_res_cam(x)

    def _compute_hi_res_cam(self, y):
        cams = torch.zeros(*y.shape, *self.feature_rect.shape[2:])
        for img_idx in range(y.shape[0]):
            for class_idx in range(y.shape[1]):
                (y[img_idx, class_idx]).backward(retain_graph=True, inputs=self.feature_rect)
                cams[img_idx, class_idx] = (self.feature_rect * self.feature_grad).sum(1)[img_idx]
        return cams


if __name__ == '__main__':
    model = Model(input_shape=const.IMAGE_SHAPE).to(const.DEVICE)
    model.eval()
    print(model)

    x = torch.rand((3, *const.IMAGE_SHAPE)).to(const.DEVICE)
    y, cam = model(x)
    print(cam.shape)
