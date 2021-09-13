PATH_SPECROGRAM_HDF5_FILE = './Train/spectrogram/amp_db.h5'
MODEL = "FC"
MODEL_NAME = 'DDAE_' + MODEL
PATH_WEIGHTS = './Train/weights/' + MODEL_NAME + '.h5'
TRAINING_FROM_SCRATCH = True
OPTIMIZER = "Adam"
SAVE_PERIOD = 25
BATCH_SIZE = 100
EPOCH_NUM = 250 