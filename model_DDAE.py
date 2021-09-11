import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, concatenate, BatchNormalization, ReLU
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend
import tensorflow as tf

print(tf.__version__)

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    print("\n")
else:
    print("Please install GPU version of TF")
    print("\n")


def DDAE(pretrained_weights=None, input_size=(256, 256)):

    size_filter_in = 16
    kernel_init = 'he_normal'
    activation_layer = None
    inputs = Input(input_size)
    print("inputs:", inputs)

    DDAE1 = Dense(500,  input_dim=input_size)(inputs)
    DDAE1 = BatchNormalization()(DDAE1)
    #DDAE1 = ReLU()(DDAE1)
    #DDAE1 = LeakyReLU()(DDAE1)

    DDAE2 = Dense(500)(DDAE1)
    DDAE2 = BatchNormalization()(DDAE2)
    #DDAE2 = ReLU()(DDAE2)
    #DDAE2 = LeakyReLU()(DDAE2)

    DDAE3 = Dense(500)(DDAE2)
    DDAE3 = BatchNormalization()(DDAE3)
    DDAE3 = ReLU()(DDAE3)
    #DDAE3 = LeakyReLU()(DDAE3)

    DDAE4 = Dense(500)(DDAE3)
    DDAE4 = BatchNormalization()(DDAE4)
    #DDAE4 = ReLU()(DDAE4)
    #DDAE4 = LeakyReLU()(DDAE4)

    DDAE5 = Dense(500)(DDAE4)
    DDAE5 = BatchNormalization()(DDAE5)
    DDAE5 = ReLU()(DDAE5)
    #DDAE5 = LeakyReLU()(DDAE5)
    outputs = Dense(256)(DDAE5)
    drop = Dropout(0.5)(outputs)

    model = Model(inputs, outputs)
    model.compile(optimizer='RMSprop',
                  loss=tf.keras.losses.Huber(), metrics=['mse'])

    # model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
