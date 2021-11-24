import tensorflow as tf
import soundfile as sf
import numpy as np
import librosa
import config_params_integration as config_params
from NC_inference_plot import inference as NC_inference
import data_tools

# To load NC and DDAE models
def load_models():
    # Define a null list for DDAE models
    DDAE_model_list = []
    NC_model = tf.keras.models.load_model(config_params.NC_PATH)

    # Load model path
    for model_path in config_params.DDAE_PATHS:
        DDAE_model_list.append(tf.keras.models.load_model(model_path))

    return NC_model, DDAE_model_list


def DDAE_inference(model, stft_mag, stft_phase):
    # Temporary used
    fix_length = 16384
    X_in = data_tools.scaled_in(stft_mag)
    print("X_in:", X_in.shape)
    X_pred = model.predict(X_in)
    print("X_pred:", X_pred.shape)

    inv_sca_X_pred = data_tools.inv_scaled_ou(X_pred)
    print("inv_sca_X_pred:", inv_sca_X_pred.shape)
    X_denoise = stft_mag - inv_sca_X_pred
    print("X_denoise:", X_denoise.shape)

    audio_denoise_recons = data_tools.matrix_spectrogram_to_numpy_audio(
        X_denoise, stft_phase, fix_length, config_params.PATH_DIR_TEST_IMAGE_DENOISE)

    sf.write(f'{config_params.PATH_DIR_PREDICT_ROOT}/{config_params.PATH_PREDICT_OUTPUT_NAME}.wav',
             audio_denoise_recons, config_params.SAMPLE_RATE, 'PCM_24')

    print(f'{config_params.PATH_DIR_PREDICT_ROOT}/{config_params.PATH_PREDICT_OUTPUT_NAME}.wav')

    data_tools.scale_dB([f'{config_params.PATH_DIR_PREDICT_ROOT}/{config_params.PATH_PREDICT_OUTPUT_NAME}.wav'],
             config_params.PATH_DIR_PREDICT_ROOT, purepath=True)


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    print(w % ncols)
    print(ncols)
    # assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    # assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr[:, :w-w % ncols].reshape(h//nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


if __name__ == "__main__":
    # Initial state of models before loading them
    NC_model, DDAE_model_list = load_models()

    print(NC_model)

    print("Loading testing audio...")
    sound, sr = sf.read(
        'Integration/test_wav/Noisy/-10_noisy.wav', dtype='float32')
    # 'test_wav/noisy_voice_long.wav', dtype='float32')

    print("Noise classifying...")
    # Load 1 sec at most
    if (sound.shape[0] >= 16384):
        result, stft_mag, stft_phase = NC_inference(
            # NC_model, sound[:16384], sr)
            NC_model, sound, sr)
    else:
        result, stft_mag, stft_phase = NC_inference(
            NC_model, sound, sr)

    index = np.argmax(result)

    if (config_params.DEV == True):
        print(result, stft_mag.shape, stft_phase.shape)
        if index == 0:
            print("Household_Appliance")
        elif index == 1:
            print("Vechicles")
        elif index == 2:
            print("Verbal_Human")

    stft_mag_db = librosa.amplitude_to_db(
        stft_mag, ref=np.max)

    stft_mag_db = blockshaped(stft_mag_db, 129, 126)
    stft_phase = blockshaped(stft_phase, 129, 126)

    DDAE_inference(DDAE_model_list[index], stft_mag_db, stft_phase)
