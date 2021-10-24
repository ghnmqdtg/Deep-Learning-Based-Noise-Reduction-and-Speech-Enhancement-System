import librosa
from numba.core import config
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
from data_tools import scale_dB, set_snr, audio_files_add_to_numpy, audio_files_to_numpy, blend_noise_voice, split_into_one_second
from data_tools import numpy_audio_to_matrix_spectrogram, scaled_in, inv_scaled_ou, matrix_spectrogram_to_numpy_audio
import os
import numpy as np
import soundfile as sf
import config_params


def createpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


def prediction():

    createpath(config_params.PATH_DIR_TEST_NOISE_dB)
    createpath(config_params.PATH_DIR_TEST_VOICE_dB)
    createpath(config_params.PATH_DIR_PREDICT_SNR_BASED)
    createpath(config_params.PATH_DIR_SAVE_NOISY)
    createpath(config_params.PATH_DIR_SAVE_IMAGE_NOISY)
    createpath(config_params.PATH_DIR_SAVE_IMAGE_VOICE)
    createpath(config_params.PATH_DIR_SAVE_IMAGE_DENOISE)

    list_noise_files = []
    list_voice_files = []
    list_noise_snr_files = []

    print("Loading...")
    # Load noise file paths
    for root, dirs, files in os.walk(config_params.PATH_DIR_TEST_NOISE_PREDICT):
        if len(files) == 0:
            continue

        for f in files:
            files = os.path.join(root, f)
            list_noise_files.append(files)

    # Load voice filenames
    for root, dirs, files in os.walk(config_params.PATH_DIR_TEST_VOICE_PREDICT):
        if len(files) == 0:
            continue

        for f in files:
            files = os.path.join(root, f)
            list_voice_files.append(files)

    target_dBFS = -30.0
    list_dB_noise_file = scale_dB(
        list_noise_files, config_params.PATH_DIR_TEST_NOISE_dB, target_dBFS)
    list_dB_voice_file = scale_dB(
        list_voice_files, config_params.PATH_DIR_TEST_VOICE_dB, target_dBFS)

    list_noise_files.clear()
    list_voice_files.clear()

    print("Setting snr...")
    # SNR = [15, 10, 0, -10, -20]
    SNR = [15, 10, 5, 0, -5, -10]
    for snr in SNR:
        createpath(config_params.PATH_DIR_SAVE_SPLIT_NOISY + str(snr) + '/')
        createpath(config_params.PATH_DIR_SAVE_SPLIT_VOICE + str(snr) + '/')
        snr_base_noise_file = set_snr(
            list_dB_voice_file, list_dB_noise_file, snr, config_params.PATH_DIR_PREDICT_SNR_BASED, config_params.SAMPLE_RATE)
        list_noise_snr_files.append(snr_base_noise_file)

    loaded_model = load_model(config_params.PATH_WEIGHTS)
    # TFLITE
    # loaded_model = tf.lite.Interpreter(config_params.PATH_WEIGHTS)
    # loaded_model = interpreter.get_signature_runner()
    for snr_noise_data in list_noise_snr_files:

        snr = snr_noise_data.split('/')[-1].split('_')[0]
        print('snr:', snr)

        noise = audio_files_to_numpy(snr_noise_data, config_params.SAMPLE_RATE)

        voice = audio_files_add_to_numpy(
            list_dB_voice_file, config_params.SAMPLE_RATE)

        print("Blending...")
        prod_voice, prod_noise, prod_noisy_voice = blend_noise_voice(voice, noise)

        sf.write(config_params.PATH_DIR_SAVE_NOISY + str(snr) + '_noisy.wav',
                 prod_noise, config_params.SAMPLE_RATE, 'PCM_24')
        sf.write(config_params.PATH_DIR_SAVE_NOISY + str(snr) + '_voice.wav',
                 prod_voice, config_params.SAMPLE_RATE, 'PCM_24')
        sf.write(config_params.PATH_DIR_SAVE_NOISY + str(snr) + '_noisy_voice_long.wav',
                 prod_noisy_voice, config_params.SAMPLE_RATE, 'PCM_24')

        cropped_list_noisy = []
        cropped_list_voice = []

        print("Cropping raw data...")
        cropped_list_noisy.extend(split_into_one_second(
            prod_noisy_voice, config_params.PATH_DIR_SAVE_SPLIT_NOISY, config_params.SAMPLE_RATE, snr, False))
        cropped_list_voice.extend(split_into_one_second(
            prod_voice, config_params.PATH_DIR_SAVE_SPLIT_VOICE, config_params.SAMPLE_RATE, snr, False))

        dim_square_spec = int(config_params.N_FFT / 2) + 1

        cropped_array_noisy = np.array(cropped_list_noisy)
        cropped_array_voice = np.array(cropped_list_voice)

        # print(cropped_list_voice)
        # print(cropped_array_voice)
        # sf.write(config_params.PATH_DIR_PREDICT_ROOT + 'Clean.wav',
        #          cropped_array_voice, sample_rate, 'PCM_24')

        print("cropped_array_noisy:", cropped_array_noisy.shape)
        print("cropped_array_voice:", cropped_array_voice.shape)
        fix_length = cropped_array_noisy.shape[1]
        print(config_params.HOP_LENGTH_FFT, fix_length)

        m_amp_db_voice,  m_pha_voice = numpy_audio_to_matrix_spectrogram(
            cropped_array_voice, dim_square_spec, config_params.N_FFT, config_params.HOP_LENGTH_FFT, config_params.PATH_DIR_SAVE_IMAGE_VOICE)

        m_amp_db_noisy_voice,  m_pha_noisy_voice = numpy_audio_to_matrix_spectrogram(
            cropped_array_noisy, dim_square_spec, config_params.N_FFT, config_params.HOP_LENGTH_FFT, config_params.PATH_DIR_SAVE_IMAGE_NOISY)

        X_in = scaled_in(m_amp_db_noisy_voice)
        print("X_in:", X_in.shape)
        X_pred = loaded_model.predict(X_in)
        print("X_pred:", X_pred.shape)

        inv_sca_X_pred = inv_scaled_ou(X_pred)
        print("inv_sca_X_pred:", inv_sca_X_pred.shape)
        X_denoise = m_amp_db_noisy_voice - inv_sca_X_pred
        print("X_denoise:", X_denoise.shape)

        audio_denoise_recons = matrix_spectrogram_to_numpy_audio(
            X_denoise, m_pha_noisy_voice, config_params.HOP_LENGTH_FFT, fix_length, config_params.PATH_DIR_SAVE_IMAGE_DENOISE)
        config_params.NB_SAMPLES = audio_denoise_recons.shape[0]

        #denoise_long = audio_denoise_recons.reshape(1, config_params.NB_SAMPLES * config_params.FRAME_SIZE)*10
        sf.write(config_params.PATH_DIR_PREDICT_ROOT + snr + '_' + config_params.PATH_PREDICT_OUTPUT_NAME,
                 audio_denoise_recons, config_params.SAMPLE_RATE, 'PCM_24')

        output_audio_list = []
        output_audio_list.append(
            config_params.PATH_DIR_PREDICT_ROOT + snr + '_' + config_params.PATH_PREDICT_OUTPUT_NAME)
        scale_dB(output_audio_list,
                 config_params.PATH_DIR_PREDICT_ROOT, target_dBFS)
        print("\n")


if __name__ == '__main__':
    prediction()
