#!/usr/bin/env python3

from ..dataset import get_generators
from .loss import CAMLoss
from .arch import Model
from .. import const
import torch
import sys

def fit(model, optimizer, losses, train, val):
    training_loss = []
    validation_loss = []
    interval = max(1, (const.EPOCHS // 10))
    for epoch in range(const.EPOCHS):
        if not (epoch+1) % interval: print('-' * 10)
        training_loss.append([])
        validation_loss.append([])

        for train_batch, val_batch in train, val:
            optimizer.zero_grad()

            X, y = map(lambda x: x.to(const.DEVICE), train_batch)
            y_pred = model(X)

            batch_loss = list(map(lambda weight, loss, pred: weight * loss(pred, y), const.LOSS_WEIGHTS, losses, y_pred))
            training_loss[-1].append(batch_loss)
            batch_loss.backward()

            optimizer.step()

        # validation loss
        if not (epoch+1) % interval: print(f'Epoch: {epoch+1}\tLoss: {sum(training_loss[-1]) / len(train)}')
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
