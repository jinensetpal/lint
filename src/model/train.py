#!/usr/bin/env python3

from ..dataset import get_generators
from .loss import CAMLoss
from .arch import Model
from .. import const
import mlflow
import torch
import sys


def fit(model, optimizer, losses, train, val):
    if const.LOG_REMOTE: mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH'])})
        mlflow.log_param('optimizer_fn', 'Stochastic Gradient Descent')

        interval = max(1, (const.EPOCHS // 10))
        for epoch in range(const.EPOCHS):
            if not (epoch+1) % interval: print('-' * 10)
            training_loss = torch.empty(len(losses))
            validation_loss = torch.empty(len(losses))
            training_acc = torch.empty(1)
            validation_acc = torch.empty(1)

            for train_batch, valid_batch in zip(train, val):
                optimizer.zero_grad()

                X_train, y_train, X_valid, y_valid = map(lambda x: x.to(const.DEVICE), [*train_batch, *valid_batch])
                y_pred_train = model(X_train)

                with torch.no_grad():
                    y_pred_valid = model(X_valid)
                training_acc = torch.vstack([training_acc.to(const.DEVICE), y_train == (y_pred_train[0] > 0.5)])
                validation_acc = torch.vstack([validation_acc.to(const.DEVICE), y_valid == (y_pred_valid[0] > 0.5)])

                train_batch_loss = list(map(lambda loss, pred: loss(pred, y_train), losses, y_pred_train))
                training_loss = torch.vstack([training_loss, torch.tensor(train_batch_loss)])
                validation_loss = torch.vstack([validation_loss, torch.tensor(list(map(lambda loss, pred: loss(pred, y_valid), losses, y_pred_valid)))])

                sum(map(lambda weight, loss: weight * loss, const.LOSS_WEIGHTS, train_batch_loss)).backward()
                optimizer.step()

            training_acc = training_acc[1:]
            validation_acc = validation_acc[1:]
            training_loss = training_loss[1:].mean(dim=0)
            validation_loss = validation_loss[1:].mean(dim=0)
            metrics = {'combined_loss': training_loss.sum().item(),
                       'bce_loss': training_loss[0].item(),
                       'cam_loss': training_loss[1].item(),
                       'train_acc': (training_acc[1:].sum() / training_acc.shape[0]).item(),
                       'valid_acc': (validation_acc[1:].sum() / validation_acc.shape[0]).item(),
                       'val_loss': validation_loss.sum().item(),
                       'val_bce_loss': validation_loss[0].item(),
                       'val_cam_loss': validation_loss[1].item()}
            mlflow.log_metrics(metrics, step=epoch)
            if not (epoch+1) % interval:
                print(f'epoch\t\t\t: {epoch+1}')
                for key in metrics: print(f'{key}\t\t: {metrics[key]}')
        print('-' * 10)


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME
    train, val, test = get_generators()

    model = Model(const.IMAGE_SHAPE).to(const.DEVICE)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=const.LEARNING_RATE,
                                momentum=const.MOMENTUM)
    losses = [torch.nn.BCELoss(), CAMLoss(model.penultimate.shape[-2:])]
    fit(model, optimizer, losses, train, val)
    torch.save(model.state_dict(), const.SAVE_MODEL_PATH / f'{name}.pt')
