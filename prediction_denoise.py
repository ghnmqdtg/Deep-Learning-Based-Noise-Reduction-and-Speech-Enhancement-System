from tensorflow.keras.models import load_model
import data_tools
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
    createpath(config_params.PATH_DIR_TEST_SNR_BASED)
    createpath(config_params.PATH_DIR_TEST_NOISY)
    createpath(config_params.PATH_DIR_TEST_IMAGE_NOISY)
    createpath(config_params.PATH_DIR_TEST_IMAGE_VOICE)
    createpath(config_params.PATH_DIR_TEST_IMAGE_DENOISE)

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

    list_dB_noise_file = data_tools.scale_dB(
        list_noise_files, config_params.PATH_DIR_TEST_NOISE_dB)
    list_dB_voice_file = data_tools.scale_dB(
        list_voice_files, config_params.PATH_DIR_TEST_VOICE_dB)

    list_noise_files.clear()
    list_voice_files.clear()

    print("Setting snr...")
    SNR = [15, 10, 5, 0, -5, -10]
    for snr in SNR:
        createpath(config_params.PATH_DIR_TEST_SPLIT_NOISY + str(snr) + '/')
        createpath(config_params.PATH_DIR_TEST_SPLIT_VOICE + str(snr) + '/')
        snr_base_noise_file = data_tools.set_snr(
            list_dB_voice_file, list_dB_noise_file, snr, config_params.PATH_DIR_TEST_SNR_BASED)
        list_noise_snr_files.append(snr_base_noise_file)

    print(f'Loading Model: {config_params.PATH_WEIGHTS}')
    loaded_model = load_model(config_params.PATH_WEIGHTS)

    for snr_noise_data in list_noise_snr_files:
        cropped_list_noisy = []
        output_audio_list = []
        snr = snr_noise_data.split('/')[-1].split('_')[0]
        print('SNR:', snr)

        noise = data_tools.audio_files_to_numpy(snr_noise_data)
        voice = data_tools.audio_files_add_to_numpy(list_dB_voice_file)

        print("Mixing...")
        prod_voice, prod_noise, prod_noisy = data_tools.mix_noise_voice(
            voice, noise)

        # sf.write(f'{config_params.PATH_DIR_TEST_NOISY}/{snr}_voice.wav',
        #          prod_voice, config_params.SAMPLE_RATE, 'PCM_24')
        # sf.write(f'{config_params.PATH_DIR_TEST_NOISY}/{snr}_noise.wav',
        #          prod_noise, config_params.SAMPLE_RATE, 'PCM_24')
        sf.write(f'{config_params.PATH_DIR_TEST_NOISY}/{snr}_noisy.wav',
                 prod_noisy, config_params.SAMPLE_RATE, 'PCM_24')

        print("Cropping raw data...")
        cropped_list_noisy.extend(data_tools.split_into_one_second(
            prod_noisy, config_params.PATH_DIR_TEST_SPLIT_NOISY, snr, False))

        cropped_array_noisy = np.array(cropped_list_noisy)

        print("cropped_array_noisy:", cropped_array_noisy.shape)
        fix_length = cropped_array_noisy.shape[1]
        print(config_params.HOP_LENGTH_FFT, fix_length)

        mag_amp_db_noisy,  mag_phase_noisy = data_tools.numpy_audio_to_matrix_spectrogram(
            cropped_array_noisy, config_params.PATH_DIR_TEST_IMAGE_NOISY)

        X_in = data_tools.scaled_in(mag_amp_db_noisy)
        X_pred = loaded_model.predict(X_in)
        inv_sca_X_pred = data_tools.inv_scaled_ou(X_pred)
        X_denoise = mag_amp_db_noisy - inv_sca_X_pred

        print("X_in:", X_in.shape)
        print("X_pred:", X_pred.shape)
        print("inv_sca_X_pred:", inv_sca_X_pred.shape)
        print("X_denoise:", X_denoise.shape)

        audio_denoise_recons = data_tools.matrix_spectrogram_to_numpy_audio(
            X_denoise, mag_phase_noisy, fix_length, config_params.PATH_DIR_TEST_IMAGE_DENOISE)

        sf.write(
            f'{config_params.PATH_DIR_PREDICT_ROOT}/{snr}_{config_params.PATH_PREDICT_OUTPUT_NAME}', audio_denoise_recons, config_params.SAMPLE_RATE, 'PCM_24')

        output_audio_list.append(
            f'{config_params.PATH_DIR_PREDICT_ROOT}/{snr}_{config_params.PATH_PREDICT_OUTPUT_NAME}')

        data_tools.scale_dB(output_audio_list, config_params.PATH_DIR_PREDICT_ROOT, purepath=True)

        print("")


if __name__ == '__main__':
    prediction()
