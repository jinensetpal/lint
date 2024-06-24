#!/usr/bin/env python3

from ..data.waterbirds import Dataset, get_generators
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .arch import Model
from src import const
import random
import torch
import sys


def group_accuracy(model, gen):
    grp = {}
    for (x, y, p) in gen:
        y = y.to(const.DEVICE)
        p = p.to(const.DEVICE)
        pred = torch.argmax(model(x.to(const.DEVICE))[0], dim=1)

        for label in [0, 1]:
            for place in [0, 1]:
                if label not in grp: grp[label] = {}
                if place not in grp[label]: grp[label][place] = torch.tensor([]).to(const.DEVICE)
                grp[label][place] = torch.cat([grp[label][place], pred[(label == torch.argmax(y, dim=1)).ravel() & (place == p)].ravel()], dim=0)

    for label in [0, 1]:
        grp[const.ENCODINGS['label'][label]] = {}
        for place in [0, 1]:
            grp[const.ENCODINGS['label'][label]][const.ENCODINGS['place'][place]] = (grp[label][place] == label).to(torch.float).mean().item()
        del grp[label]
    return grp


def visualize(model, gen):
    fig = plt.figure(figsize=(14, 14),
                     facecolor='white')

    for idx, sample in enumerate(random.sample(range(len(gen)), 16)):
        X, y = gen[sample]
        y_pred, cam = model(X.unsqueeze(0).to(const.DEVICE))
        cam = cam[0][y.argmax().item()].detach().abs()
        cam -= cam.min()
        cam /= cam.max()
        fig.add_subplot(4, 4, idx + 1)
        buf = 'Predicted Class = ' + str(y_pred.argmax().item())
        plt.xlabel(buf)
        plt.imshow(X.permute(1, 2, 0).detach().numpy(), alpha=0.5)
        plt.imshow(F.interpolate(cam[None, None, ...], const.IMAGE_SIZE, mode='bilinear')[0][0].numpy(), cmap='jet', alpha=0.5)

    plt.tight_layout()
    plt.show()
    fig.savefig(const.DATA_DIR / 'evals' / f'{model.name}.png')


if __name__ == '__main__':
    name = sys.argv[2] if len(sys.argv) > 2 else const.MODEL_NAME

    model = Model(input_shape=const.IMAGE_SHAPE)
    model.load_state_dict(torch.load(const.SAVE_MODEL_PATH / f'{name}.pt', map_location=const.DEVICE))
    model.name = name
    model.eval()

    if sys.argv[1] == 'group':
        with open(const.DATA_DIR / 'evals' / f'{model.name}.txt', 'w') as f:
            for gen in get_generators(state='evaluation'): f.write(gen.dataset.split + ': ' + str(group_accuracy(model, gen)) + '\n')
    else: visualize(model, Dataset(const.DATA_DIR / 'waterbirds' / 'metadata.csv', split='test'))
