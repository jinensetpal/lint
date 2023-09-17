#!/usr/bin/env python3

from .. import const
from torch import nn
import torchvision
import torch

class Model(torch.nn.Module):
    def __init__(self, input_shape, show_cams=True):
        super().__init__()
        self.show_cams = show_cams

        self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()

        if self.show_cams:
            self.backbone.layer4[-1].relu.register_forward_hook(self._hook)
            self.penultimate = None

        self.linear = nn.Linear(self._backbone_out_shape(input_shape), 2)
        self.softmax = nn.Softmax(dim=1)  # ~equivalent to sigmoid since classes = 2; for CAMs

    def forward(self, x):
        x = self.backbone(x)
        x = self.linear(x)
        x = self.softmax(x)
        
        if self.show_cams: return x, self._compute_cam(torch.argmax(x, dim=1))
        return x

    def _backbone_out_shape(self, input_shape):
        with torch.no_grad():
            return self.backbone(torch.rand((1, *input_shape))).shape[-1]

    def _hook(self, model, i, o):
        self.penultimate = o.detach()

    def _compute_cam(self, label_index):
        class_weights = self.get_submodule('linear').weight
        def single_cam(penultimate):
            return torch.vmap(lambda x: torch.tensordot(x, penultimate, dims=1), in_dims=0)(class_weights)

        return torch.vmap(single_cam, in_dims=0)(self.penultimate)

if __name__ == '__main__':
    model = Model(input_shape=const.IMAGE_SHAPE, show_cams=const.SHOW_CAMS)
    print(model)

    x = torch.rand((3, *const.IMAGE_SHAPE))
    print(model(x))
