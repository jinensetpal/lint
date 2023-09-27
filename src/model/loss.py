#!/usr/bin/env python3

from astropy.convolution import Gaussian2DKernel
from ..clustering.train import get_model
from ..data.siamese import Dataset
import torch.nn as nn
from .. import const
import numpy as np
import torchvision
import torch


class EmbeddingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = get_model().to(const.DEVICE)
        self.encoder.load_state_dict(torch.load(const.SAVE_MODEL_PATH / 'maskencoder.pt',
                                     map_location=const.DEVICE))
        self.encoder.eval()
        self.means = Dataset().means(self.encoder)

        self.d_neg = (self.means[1] - self.means[0]).pow(2).sum(0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, positive, _):
        with torch.no_grad():
            d_pos = (self.encoder(torchvision.transforms.functional.resize(self.sigmoid(positive) > .5, const.IMAGE_SIZE, antialias=True).to(torch.float).repeat(1, 3, 1, 1)) - self.means[0]).pow(2).sum(1)

        return nn.functional.relu(d_pos - self.d_neg).mean().item()


class RadialLoss(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.kernel = torch.tensor(np.array(Gaussian2DKernel(const.IMAGE_SIZE[0] * .01, x_size=shape[0], y_size=shape[1]))).to(const.DEVICE)
        self.kernel -= self.kernel.max()
        self.kernel = torch.square(self.kernel)

    def forward(self, y_pred, _):
        return torch.square(y_pred[:, 0] * self.kernel).mean()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    loss = RadialLoss((7, 7))
    plt.imshow(loss.kernel.detach().cpu())
    plt.colorbar()
    plt.show()
