# this is the run file.

import keras
import keras.backend as K
import tensorflow
import math
import os

from data_generator import Cifar10Gen
from GoogLeNet import googlenet

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

path = '../database/cifar10/'
num_classes = 10
epochs = 50
batch_size = 128
learning_rate = 0.01
steps = math.ceil(10000 / batch_size)

gen = Cifar10Gen(path=path, batch_size=batch_size)
model = googlenet(input_shape=(224, 224, 3,), num_classes=num_classes)

opt = keras.optimizers.SGD(learning_rate=0, momentum=0.9)


def step_decay(epoch):
    drop = 0.04
    times = math.floor(epoch / 8)
    new_lr = learning_rate * math.pow(1 - drop, times)
    return new_lr


callback = keras.callbacks.LearningRateScheduler(step_decay, verbose=1)
model.compile(optimizer=opt,
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
model.fit_generator(gen.train_generator(),
                    steps_per_epoch=5*steps,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[callback],
                    validation_data=gen.valid_generator(),
                    validation_steps=steps)
print()
print('*****saving model****')
keras.models.save_model(model, './checkpoint/googlenet_checkpoint_{}.h5'.format(epochs))