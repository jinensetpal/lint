#!/usr/bin/env python3

from pathlib import PosixPath
from src import const
import pandas as pd
import torchvision
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, split=None, state='training'):
        if isinstance(df, PosixPath): df = pd.read_csv(df)

        if split:
            self.df = df[df['split'] == const.ENCODINGS['split'].index(split)].reset_index()
            self.split = split
        else:
            self.df = df
            self.split = 'all'

        self.state = state

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = torchvision.transforms.functional.resize(torchvision.io.read_image((const.DATA_DIR / self.df['img_filename'].iloc[idx]).as_posix()), const.IMAGE_SIZE, antialias=True)
        X = X / 255  # normalization
        y = torch.zeros((const.N_CLASSES))
        y[self.df['y'][idx]] = 1

        if self.state == 'evaluation': return X, y, torch.tensor(self.df['place'][idx])
        return X, y


def get_generators(state='training'):
    return [torch.utils.data.DataLoader(Dataset(const.DATA_DIR / 'metadata.csv', split=split, state=state),
                                        batch_size=const.BATCH_SIZE, shuffle=True) for split in const.ENCODINGS['split']]


if __name__ == '__main__':
    print(Dataset(const.DATA_DIR / 'metadata.csv')[0])
