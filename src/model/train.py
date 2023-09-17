#!/usr/bin/env python3

from tensorflow.keras import backend as K
from ..data.generator import get_dataset
from .callbacks import get_callbacks
from .arch import get_model
from .loss import CAMLoss
import tensorflow as tf
from .. import const
import mlflow
import sys
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, optimizer, loss, data, val):
    training_loss = []
    for epoch in range(const.EPOCHS):
        if not (epoch+1) % (const.EPOCHS // 10): print('-' * 10)
        training_loss.append([])

        for batch in data:
            optimizer.zero_grad()

            X, y = batch
            y_pred = model(X.to(device))

            batch_loss = loss(y_pred, y.to(device))
            batch_loss.backward()

            optimizer.step()

            training_loss[-1].append(batch_loss)
        # validation loss
        if not (epoch+1) % (const.EPOCHS // 10): print(f'Epoch: {epoch+1}\tLoss: {sum(training_loss[-1]) / const.BATCH_SIZE}')
    print('-' * 10)


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME
    multiheaded = const.MODEL_NAME != name

    train, val, test = get_dataset()
    model = get_model(const.IMAGE_SIZE, const.N_CLASSES, name, const.N_CHANNELS, multiheaded=multiheaded)
    model.summary()

    optimizer = tf.keras.optimizers.SGD(learning_rate=const.LEARNING_RATE[0],
                                        momentum=const.MOMENTUM)
    loss_weights = const.LOSS_WEIGHTS[0] if multiheaded else K.variable(1)
    losses = ['binary_crossentropy', CAMLoss(loss_weights)] if multiheaded else 'binary_crossentropy'
    model.compile(optimizer=optimizer,
                  loss=losses,
                  loss_weights=loss_weights,
                  metrics={'output': 'accuracy'})

    if const.LOG: mlflow.tensorflow.autolog(log_models=False)
    model.fit(train,
              epochs=const.EPOCHS,
              validation_data=val,
              use_multiprocessing=False,
              callbacks=get_callbacks(const.LIMIT, loss_weights))

    model.compile(optimizer=optimizer,
                  loss=losses,
                  metrics={'output': 'accuracy'})  # recompiling since tensorflow does not serialize backend-tampered variables
    metrics = model.evaluate(test)
    model.save(os.path.join(const.BASE_DIR, *const.SAVED_MODEL_PATH, name))
#!/usr/bin/env python3

from ..dataset import get_generators
from .arch import Model
import const
import torch

if __name__ == '__main__':
    train, val, test = get_generators()

    model = Model(const.IMAGE_SHAPE).to(device)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=const.LEARNING_RATE,
                                momentum=const.MOMENTUM)
    loss = torch.nn.CrossEntropyLoss()
    train(model, optimizer, loss, train, val)
    torch.save(model, const.SAVE_MODEL_PATH / 'model.pt')

    embed()
