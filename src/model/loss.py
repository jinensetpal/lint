#!/usr/bin/env python3

from astropy.convolution import Gaussian2DKernel
import torch.nn as nn
from .. import const
import numpy as np
import torch


class CAMLoss(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.kernel = torch.tensor(np.array(Gaussian2DKernel(const.IMAGE_SIZE[0] * .01, x_size=shape[0], y_size=shape[1]))).to(const.DEVICE)
        self.kernel -= self.kernel.max()
        self.kernel = torch.square(self.kernel)

    def forward(self, y_pred, _):
        return torch.square(y_pred[:, 0] * self.kernel).mean()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    loss = CAMLoss((7, 7))
    plt.imshow(loss.kernel.detach().cpu())
    plt.colorbar()
    plt.show()
