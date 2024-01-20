#!/usr/bin/env python3

from ..data.waterbirds import get_generators
from ..arch import Model
from src import const
import torch
import sys


def group_accuracy(model, gen):
    grp = {}
    for (x, y, p) in gen:
        y = y.to(const.DEVICE)
        p = p.to(const.DEVICE)
        with torch.no_grad():
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


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME

    model = Model(input_shape=const.IMAGE_SHAPE).to(const.DEVICE)
    model.load_state_dict(torch.load(const.SAVE_MODEL_PATH / f'{name}.pt', map_location=const.DEVICE))
    model.eval()

    for gen in get_generators(state='evaluation'): print(gen.dataset.split, group_accuracy(model, gen))
