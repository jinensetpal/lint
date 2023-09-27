#!/usr/bin/env python3

from torch import nn


class TripletLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, anchor, positive, negative):
        d_pos = (positive - anchor).pow(2).sum(1)
        d_neg = (negative - anchor).pow(2).sum(1)

        return nn.functional.relu(d_pos + self.alpha - d_neg).mean()
