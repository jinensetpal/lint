#!/usr/bin/env python3

from torch.nn import functional as F
from torch.nn import Module
from src import const
import torch


def l1_penalty(self):
    return const.S_L1_ALPHA * sum([layer.abs().sum() for layer in self.parameters()])


class TripletLoss(Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, anchor, positive, negative):
        d_pos = (positive - anchor).pow(2).sum(1)
        d_neg = (negative - anchor).pow(2).sum(1)

        return F.relu(d_pos + self.alpha - d_neg).mean()
