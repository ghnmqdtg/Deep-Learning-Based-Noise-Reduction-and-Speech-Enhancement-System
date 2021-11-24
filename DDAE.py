import config_params
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
# from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Activation
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.layers import GRU, BatchNormalization, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_memory_growth(gpus[0], True)


def DDAE(input_size=(129, 126)):
    if (config_params.MODEL == "FC"):
        model = Sequential([
            InputLayer(input_size),
            Dense(500),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Dense(500),
            BatchNormalization(),
            # Activation('relu'),
            Dropout(0.1),
            Dense(500),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Dense(500),
            BatchNormalization(),
            # Activation('relu'),
            Dropout(0.1),
            Dense(500),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.1),
            Dense(126),
            # BatchNormalization(),
            # Activation('relu'),
            # Dropout(0.5),
        ], name='DDAE_FC')
    elif (config_params.MODEL == "GRU"):
        model = Sequential([
            InputLayer(input_size),
            GRU(126, dropout=0.1, return_sequences=True),
            BatchNormalization(),
            GRU(126, dropout=0.1, return_sequences=True),
            BatchNormalization(),
            GRU(63, dropout=0.1, return_sequences=True),
            BatchNormalization(),
            GRU(63, dropout=0.1, return_sequences=True),
            BatchNormalization(),
            GRU(32, dropout=0.1, return_sequences=True),
            BatchNormalization(),
            GRU(32, dropout=0.1, return_sequences=True),
            BatchNormalization(),
            GRU(63, dropout=0.1, return_sequences=True),
            BatchNormalization(),
            GRU(63, dropout=0.1, return_sequences=True),
            BatchNormalization(),
            GRU(126, dropout=0.1, return_sequences=True),
            BatchNormalization(),
            GRU(126, dropout=0.1, return_sequences=True),
            BatchNormalization(),
        ], name='DDAE_GRU')

    # Set optimizer
    initial_learning_rate = 0.1
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=200000,
        decay_rate=0.96,
        staircase=True)
    if (config_params.OPTIMIZER == "Adam"):
        optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                         # optimizer = Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999,
                         epsilon=1e-07, decay=0.0, amsgrad=True, name='Adam')
    elif (config_params.OPTIMIZER == "RMSprop"):
        optimizer = RMSprop(learning_rate=lr_schedule)

    model.compile(loss=Huber(),
                  optimizer=optimizer,
                  metrics=['mse'])

    if not config_params.TRAINING_FROM_SCRATCH:
        model.load_weights(config_params.PATH_WEIGHTS)

    return model


class Train():

    @staticmethod
    def load_data():
        '''
        Load the dataset
            It's an early version of load_data. This dataset consists of x_train,
            y_train, x_test and y_test, four `.npy` files. We will save
            the dataset into `.npy` file.
        '''

        hdf5_file = config_params.PATH_SPECROGRAM_HDF5_FILE

        x_train = tfio.IODataset.from_hdf5(
            hdf5_file, dataset="/trainnoise")
        y_train = tfio.IODataset.from_hdf5(
            hdf5_file, dataset="/trainclean")
        x_test = tfio.IODataset.from_hdf5(
            hdf5_file, dataset="/valnoise")
        y_test = tfio.IODataset.from_hdf5(
            hdf5_file, dataset="/valclean")

        train = tf.data.Dataset.zip((x_train, y_train)).batch(
            config_params.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        test = tf.data.Dataset.zip((x_test, y_test)).batch(
            config_params.BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

        return train, test


if __name__ == '__main__':

    train, test = Train().load_data()

    model = DDAE((129, 126))
    model.summary()

    # Set checkpoint
    checkpoint = ModelCheckpoint(config_params.PATH_WEIGHTS, verbose=2,
                                 monitor='val_loss', save_best_only=True, mode='auto')

    # Set up EarlyStopping and checkpoint
    # STEPS_PER_EPOCH = x_train.shape[0] // config_params.BATCH_SIZE
    earlytopping = EarlyStopping(monitor='loss', patience=10)
    checkpoint = ModelCheckpoint(filepath=config_params.PATH_WEIGHTS,
                                 monitor='val_loss',
                                 verbose=True,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 #  save_freq=config_params.SAVE_PERIOD * STEPS_PER_EPOCH,
                                 mode='auto')

    history = model.fit(train,
                        validation_split=0,
                        epochs=config_params.EPOCH_NUM,
                        # steps_per_epoch=STEPS_PER_EPOCH,
                        batch_size=config_params.BATCH_SIZE,
                        verbose=1,
                        callbacks=[checkpoint],
                        # callbacks=[earlytopping, checkpoint],
                        # callbacks=[earlytopping],
                        validation_data=test,
                        shuffle=True)

    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')
    mse = history.history.get('mse')
    val_mse = history.history.get('val_mse')

    plt.figure(0)
    plt.subplot(121)
    plt.plot(range(len(loss)), loss, label='Training')
    plt.plot(range(len(val_loss)), val_loss, label='Validation')
    plt.title('Huber Loss')
    plt.legend(loc='best')
    plt.subplot(122)
    plt.plot(range(len(mse)), mse, label='Training')
    plt.plot(range(len(val_mse)), val_mse, label='Validation')
    plt.title('MSE Loss')
    plt.legend(loc='best')
    plt.savefig(
        config_params.PATH_CURVE, dpi=300, format='png')
    plt.close()
    print(
        f'Result saved into {config_params.PATH_CURVE}')
    plt.show()
