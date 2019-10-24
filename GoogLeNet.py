# this is the net model file.

import keras
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers import AvgPool2D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.models import Model

conv_init = keras.initializers.TruncatedNormal(0, 0.01)
fc_init = keras.initializers.TruncatedNormal(0, 0.005)


def inception(input, filter1, filter3re, filter3, filter5re, filter5, poolproj):
    conv1 = Conv2D(filter1, (1, 1),
                   padding='same',
                   activation='relu',
                   # kernel_initializer=conv_init
                   )(input)
    conv3r = Conv2D(filter3re, (1, 1),
                    padding='same',
                    activation='relu',
                    # kernel_initializer=conv_init
                    )(input)
    conv3 = Conv2D(filter3, (3, 3),
                   padding='same',
                   activation='relu',
                   # kernel_initializer=conv_init
                   )(conv3r)
    conv5r = Conv2D(filter5re, (1, 1),
                    padding='same',
                    activation='relu',
                    # kernel_initializer=conv_init
                    )(input)
    conv5 = Conv2D(filter5, (5, 5),
                   padding='same',
                   activation='relu',
                   # kernel_initializer=conv_init
                   )(conv5r)
    maxpool = MaxPool2D((3, 3),
                        strides=1,
                        padding='same'
                        )(input)
    proj = Conv2D(poolproj, (1, 1),
                  padding='same',
                  activation='relu',
                  # kernel_initializer=conv_init
                  )(maxpool)
    x = Concatenate(axis=3)([conv1, conv3, conv5, proj])
    return x


def googlenet(input_shape, num_classes):
    net = {}
    input_tensor = Input(input_shape)
    net['input'] = input_tensor

    # layer 1
    net['c1'] = Conv2D(64, (7, 7),
                       strides=2,
                       padding='same',
                       activation='relu',
                       kernel_initializer=conv_init,
                       name='c1')(net['input'])
    net['p1'] = MaxPool2D((3, 3),
                          strides=2,
                          padding='same',
                          name='p1')(net['c1'])
    net['bn1'] = BatchNormalization(name='bn1')(net['p1'])

    # layer 2
    net['c2_reduce'] = Conv2D(64, (1, 1),
                              activation='relu',
                              padding='same',
                              kernel_initializer=conv_init,
                              name='c2_reduce')(net['bn1'])
    net['c2'] = Conv2D(192, (3, 3),
                       strides=1,
                       padding='same',
                       activation='relu',
                       kernel_initializer=conv_init,
                       name='c2')(net['c2_reduce'])
    net['bn2'] = BatchNormalization(name='bn2')(net['c2'])
    net['p2'] = MaxPool2D((3, 3),
                          strides=2,
                          padding='same',
                          name='p2')(net['bn2'])

    # layer 3
    net['inception_3a'] = inception(net['p2'], 64, 96, 128, 16, 32, 32)
    net['inception_3b'] = inception(net['inception_3a'], 128, 128, 192, 32, 96, 64)
    net['p3'] = MaxPool2D((3, 3),
                          strides=2,
                          padding='same',
                          name='p3')(net['inception_3b'])

    # layer 4
    net['inception_4a'] = inception(net['p3'], 192, 96, 208, 16, 48, 64)
    net['inception_4b'] = inception(net['inception_4a'], 160, 112, 224, 24, 64, 64)
    net['inception_4c'] = inception(net['inception_4b'], 128, 128, 256, 24, 64, 64)
    net['inception_4d'] = inception(net['inception_4c'], 112, 144, 288, 32, 64, 64)
    net['inception_4e'] = inception(net['inception_4d'], 256, 160, 320, 32, 128, 128)
    net['p4'] = MaxPool2D((3, 3),
                          strides=2,
                          padding='same',
                          name='p4')(net['inception_4e'])

    # layer 5
    net['inception_5a'] = inception(net['p4'], 256, 160, 320, 32, 128, 128)
    net['inception_5b'] = inception(net['inception_5a'], 384, 192, 384, 48, 128, 128)

    net['avgpool'] = AvgPool2D((7, 7),
                               strides=1,
                               padding='same',
                               name='avgpool')(net['inception_5b'])
    net['dropout'] = Dropout(0.4, name='dropout')(net['avgpool'])
    net['flat'] = Flatten(name='flat')(net['dropout'])
    net['linear'] = Dense(num_classes,
                          activation='softmax',
                          kernel_initializer=fc_init,
                          name='linear')(net['flat'])
    net['output'] = net['linear']
    model = Model(net['input'], net['output'])
    model.summary()
    return model
