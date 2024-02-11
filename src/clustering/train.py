#!/usr/bin/env python3

from .loss import TripletLoss, l1_penalty
from ..data.siamese import Dataset
from .. import const
from torch import nn
import mlflow
import torch
import sys


def fit(model, optimizer, loss, dataloader):
    if const.LOG_REMOTE: mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH'])})
        mlflow.log_param('optimizer_fn', 'Adam')

        interval = max(1, (const.S_EPOCHS // 10))
        for epoch in range(const.S_EPOCHS):
            if not (epoch+1) % interval: print('-' * 10)
            epoch_loss = torch.empty(1)
            triplet_loss = torch.empty(1)
            l1_loss = torch.empty(1)

            for batch in dataloader:
                optimizer.zero_grad()

                anchor, positive, negative = [model(x.to(const.DEVICE)) for x in batch]
                triplet_batch_loss = loss(anchor, positive, negative)
                l1_batch_loss = l1_penalty(model)

                batch_loss = triplet_batch_loss + l1_batch_loss
                batch_loss.backward()
                optimizer.step()

                l1_loss = torch.vstack([l1_loss.to(const.DEVICE), l1_batch_loss])
                triplet_loss = torch.vstack([triplet_loss.to(const.DEVICE), triplet_batch_loss])
                epoch_loss = torch.vstack([epoch_loss.to(const.DEVICE), batch_loss])

            metrics = {'batch_loss': epoch_loss[1:].mean().item(),
                       'triplet_loss': triplet_loss[1:].mean().item(),
                       'l1_loss': l1_loss[1:].mean().item()}
            mlflow.log_metrics(metrics, step=epoch)
            if not (epoch+1) % interval:
                print(f'epoch\t\t\t: {epoch+1}')
                for key in metrics: print(f'{key}\t\t: {metrics[key]}')
        print('-' * 10)


if __name__ == '__main__':
    const.MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME

    dataloader = torch.utils.data.DataLoader(Dataset(),
                                             batch_size=const.S_BATCH_SIZE,
                                             shuffle=True)
    model = nn.Sequential(nn.Flatten(), nn.LazyLinear(128), nn.Linear(128, 1)).to(const.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=const.S_LEARNING_RATE)
    loss = TripletLoss(const.S_ALPHA).to(const.DEVICE)

    fit(model, optimizer, loss, dataloader)
    torch.save(model.state_dict(), const.SAVE_MODEL_PATH / f'{const.S_MODEL_NAME}.pt')
