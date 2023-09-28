#!/usr/bin/env python3

from ..data.siamese import Dataset
from .loss import TripletLoss
from .. import const
import torchvision
import mlflow
import torch
import sys


def get_model():
    backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    return backbone


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

            for batch in dataloader:
                optimizer.zero_grad()

                anchor, positive, negative = [model(x.to(const.DEVICE)) for x in batch]
                batch_loss = loss(anchor, positive, negative)
                batch_loss.backward()
                optimizer.step()

                epoch_loss = torch.vstack([epoch_loss.to(const.DEVICE), batch_loss])

            metrics = {'triplet_loss': epoch_loss[1:].mean().item()}
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
    model = get_model().to(const.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=const.S_LEARNING_RATE)
    loss = TripletLoss(const.S_ALPHA).to(const.DEVICE)

    fit(model, optimizer, loss, dataloader)
    torch.save(model.state_dict(), const.SAVE_MODEL_PATH / f'{const.MODEL_NAME}.pt')
