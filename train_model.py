import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import HDF5Matrix
import h5py
from model_DDAE import DDAE
from data_tools import scaled_in, scaled_ou
import os

# for tf 1.xx
import tensorflow as tf
config = tf.compat.v1.ConfigProto
# config.gpu_options.allow_growth = True
# session = tf.InteractiveSession(config=config)

# def generator(noisy, clean, batch_size):


def generator(hdf5_file, batch_size, dataset_noise, dataset_clean):

    noisy = HDF5Matrix(hdf5_file, dataset_noise)
    size = noisy.end
    clean = HDF5Matrix(hdf5_file, dataset_clean)
    idx = 0
    while True:
        last_batch = idx + batch_size > size
        end = idx + batch_size if not last_batch else size
        yield noisy[idx:end], clean[idx:end]
        idx = end if not last_batch else 0


def data_statistic(hdf5_file, train_dataset, val_dataset):

    train_x = HDF5Matrix(hdf5_file, train_dataset)
    val_x = HDF5Matrix(hdf5_file, val_dataset)
    return train_x.end, val_x.end


def createpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


def training(path_save_spectrogram, weights_path, name_model, training_from_scratch, epochs, batch_size):

    createpath(weights_path)

    '''
    X_in = np.load(path_save_spectrogram +'noisy_voice_amp_db'+".npy")
    X_ou = np.load(path_save_spectrogram +'voice_amp_db'+".npy")

    X_ou = X_in - X_ou

    print("X_in before scaled:\n",stats.describe(X_in.reshape(-1,1)))
    print("X_ou before scaled:\n",stats.describe(X_ou.reshape(-1,1)))

    X_in = scaled_in(X_in)
    X_ou = scaled_ou(X_ou)

    print("X_in shape:",X_in.shape)
    print("X_ou shape:",X_ou.shape)
    print("X_in after scaled:\n",stats.describe(X_in.reshape(-1,1)))
    print("X_ou after scaled:\n",stats.describe(X_ou.reshape(-1,1)))
    '''

    '''
    #Reshape for training
    X_in = X_in[:,:,:]
    X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
    X_ou = X_ou[:,:,:]
    X_ou = X_ou.reshape(X_ou.shape[0],X_ou.shape[1],X_ou.shape[2],1)
    
    print("X_in shape:",X_in.shape)
    print("X_ou shape:",X_ou.shape)
    '''

    hdf5_file = path_save_spectrogram + 'amp_db.h5'

    #X_train, X_test, y_train, y_test = train_test_split(X_in, X_ou, test_size=0.10, random_state=43)

    train_generator = generator(
        hdf5_file, batch_size, dataset_noise='trainnoise', dataset_clean='trainclean')
    val_generator = generator(
        hdf5_file, batch_size, dataset_noise='valnoise', dataset_clean='valclean')
    nb_train_samples, nb_val_samples = data_statistic(
        hdf5_file, 'trainnoise', 'valnoise')

    # If training from scratch
    if training_from_scratch:
        generator_nn = DDAE()
    # If training from pre-trained weights
    else:
        generator_nn = DDAE(pretrained_weights=weights_path+'model_DDAE.h5')

    checkpoint = ModelCheckpoint(weights_path+'model_DDAE.h5', verbose=2,
                                 monitor='val_loss', save_best_only=True, mode='auto')

    generator_nn.summary()
    #history = generator_nn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[checkpoint], verbose=1, validation_data=(X_test, y_test))

    history = generator_nn.fit_generator(generator=train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint],
                                         validation_data=val_generator, validation_steps=nb_val_samples // batch_size, max_queue_size=2, workers=1, use_multiprocessing=False, shuffle='batch')
    #steps_per_epoch=nb_train_samples // batch_size
    #validation_steps=nb_val_samples // batch_size

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    # plt.yscale('log')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('./lossVSepoch.png')
    plt.show()


if __name__ == '__main__':

    path_save_spectrogram = './Train/spectrogram/'
    weights_path = './Train/weights/'
    name_model = 'model_DDAE'
    training_from_scratch = True
    epochs = 250
    batch_size = 53

    training(path_save_spectrogram, weights_path, name_model,
             training_from_scratch, epochs, batch_size)
