#!/usr/bin/env python3

from ..data.waterbirds import get_generators
from .loss import RadialLoss, EmbeddingLoss
from .arch import Model
from src import const
import mlflow
import torch
import sys


def fit(model, optimizer, losses, train, val):
    if const.LOG_REMOTE: mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    best = {'param': model.state_dict(),
            'epoch': 0,
            'acc': 0.0}

    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH'])})
        mlflow.log_param('optimizer_fn', 'SGD')

        interval = max(1, (const.EPOCHS // 10))
        for epoch in range(const.EPOCHS):
            if not (epoch+1) % interval: print('-' * 10)
            train_loss = torch.empty(len(losses))
            valid_loss = torch.empty(len(losses))
            train_acc = torch.empty(1)
            valid_acc = torch.empty(1)

            for train_batch, valid_batch in zip(train, val):
                optimizer.zero_grad()

                X_train, y_train, X_valid, y_valid = map(lambda x: x.to(const.DEVICE), [*train_batch, *valid_batch])
                y_pred_train = model(X_train)
                y_pred_valid = model(X_valid)

                train_acc = torch.vstack([train_acc.to(const.DEVICE), (torch.argmax(y_train, dim=1) == torch.argmax(y_pred_train[0], dim=1)).unsqueeze(1)])
                valid_acc = torch.vstack([valid_acc.to(const.DEVICE), (torch.argmax(y_valid, dim=1) == torch.argmax(y_pred_valid[0], dim=1)).unsqueeze(1)])

                train_batch_loss = list(map(lambda loss, pred: loss(pred, y_train), losses, y_pred_train))
                train_batch_loss[-1] = min(10 * train_batch_loss[0], train_batch_loss[-1])
                train_loss = torch.vstack([train_loss, torch.tensor(train_batch_loss)])

                valid_batch_loss = list(map(lambda loss, pred: loss(pred, y_valid), losses, y_pred_valid))
                valid_batch_loss[-1] = min(10 * valid_batch_loss[0], valid_batch_loss[-1])
                valid_loss = torch.vstack([valid_loss, torch.tensor(valid_batch_loss)])

                sum(map(lambda weight, loss: weight * loss, const.LOSS_WEIGHTS, train_batch_loss)).backward()
                optimizer.step()

            train_acc = train_acc[1:]
            valid_acc = valid_acc[1:]
            train_loss = train_loss[1:].mean(dim=0)
            valid_loss = valid_loss[1:].mean(dim=0)
            metrics = {'combined_loss': train_loss.sum().item(),
                       'cse_loss': train_loss[0].item(),
                       'cam_loss': train_loss[1].item(),
                       'train_acc': (train_acc[1:].sum() / train_acc.shape[0]).item(),
                       'valid_acc': (valid_acc[1:].sum() / valid_acc.shape[0]).item(),
                       'val_loss': valid_loss.sum().item(),
                       'val_cse_loss': valid_loss[0].item(),
                       'val_cam_loss': valid_loss[1].item()}
            mlflow.log_metrics(metrics, step=epoch)

            if metrics['valid_acc'] > best['acc']:
                best['param'] = model.state_dict()
                best['epoch'] = epoch+1
                best['acc'] = metrics['valid_acc']

            if not (epoch+1) % interval:
                print(f'epoch\t\t\t: {epoch+1}')
                for key in metrics: print(f'{key}\t\t: {metrics[key]}')
        mlflow.log_param('selected_epoch', best['epoch'])
        print('-' * 10)


if __name__ == '__main__':
    const.MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME
    train, val, test = get_generators()

    model = Model(const.IMAGE_SHAPE).to(const.DEVICE)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=const.LEARNING_RATE,
                                momentum=const.MOMENTUM)
    losses = [torch.nn.CrossEntropyLoss(),
              EmbeddingLoss() if const.USE_SIAMESE_LOSS else RadialLoss(model.penultimate.shape[-2:])]
    fit(model, optimizer, losses, train, val)
    torch.save(model.state_dict(), const.SAVE_MODEL_PATH / f'{const.MODEL_NAME}.pt')
