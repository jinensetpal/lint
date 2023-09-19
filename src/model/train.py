#!/usr/bin/env python3

from ..dataset import get_generators
from .loss import CAMLoss
from .arch import Model
from .. import const
import torch
import sys

def fit(model, optimizer, losses, data, val):
    training_loss = []
    interval = max(1, (const.EPOCHS // 10))
    for epoch in range(const.EPOCHS):
        if not (epoch+1) % interval: print('-' * 10)
        training_loss.append([])

        for batch in data:
            optimizer.zero_grad()

            X, y = map(lambda x: x.to(const.DEVICE), batch)
            y_pred = model(X)

            batch_loss = sum(map(lambda weight, loss, pred: weight * loss(pred, y), const.LOSS_WEIGHTS, losses, y_pred))
            batch_loss.backward()

            optimizer.step()

            training_loss[-1].append(batch_loss)
        # validation loss
        if not (epoch+1) % interval: print(f'Epoch: {epoch+1}\tLoss: {sum(training_loss[-1]) / const.BATCH_SIZE}')
    print('-' * 10)


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME
    train, val, test = get_generators()

    model = Model(const.IMAGE_SHAPE).to(const.DEVICE)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=const.LEARNING_RATE,
                                momentum=const.MOMENTUM)
    losses = [torch.nn.CrossEntropyLoss(), CAMLoss(model.penultimate.shape[-2:])]
    fit(model, optimizer, losses, train, val)
    torch.save(model.state_dict(), const.SAVE_MODEL_PATH / f'{name}.pt')

    embed()
