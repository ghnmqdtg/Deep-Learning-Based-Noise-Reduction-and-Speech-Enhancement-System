import os
import sys
import inspect
# 
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import config_params

# Development
DEV = True

# Inference
# NC_PATH = './Integration/model_files/model_0928_3_cats_equal_samples.h5'
NC_PATH = './Integration/model_files/model_1102_4_cats.h5'
NOISE_LIST = ['Household_Appliance', 'TVnRadio', 'Vechicles', 'Verbal_Human']
DDAE_PATHS = [
    f'./Integration//model_files/DDAE_{config_params.MODEL}_{cls}.h5' for cls in NOISE_LIST]

# Save paths
PATH_DIR_SAVE_IMAGE_DENOISE = './Integration/test_wav/Denoised/images'
PATH_DIR_PREDICT_ROOT = './Integration/test_wav/Denoised/audio'
PATH_PREDICT_OUTPUT_NAME = 'denoised'

# Sync from config_params
HOP_LENGTH_FFT = config_params.HOP_LENGTH_FFT
NB_SAMPLES = config_params.NB_SAMPLES
SAMPLE_RATE = config_params.SAMPLE_RATE