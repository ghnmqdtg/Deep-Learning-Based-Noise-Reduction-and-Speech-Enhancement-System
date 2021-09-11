import librosa
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
from data_tools import scale_dB, set_snr, audio_files_add_to_numpy, audio_files_to_numpy, blend_noise_voice, split_into_one_second
from data_tools import numpy_audio_to_matrix_spectrogram, scaled_in, inv_scaled_ou, matrix_spectrogram_to_numpy_audio
import os
import numpy as np
import soundfile as sf


def createpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


def prediction(weights_path, name_model, dB_noise_dir, dB_voice_dir, snr_based_dir, splitted_noisy_dir, splitted_voice_dir, noise_dir_prediction, voice_dir_prediction, dir_save_prediction, noisy_save_prediction,
               path_save_noisy_image, path_save_voice_image, audio_output_prediction, sample_rate, min_duration, frame_length, hop_length_frame, nb_samples, n_fft, hop_length_fft):

    createpath(dB_noise_dir)
    createpath(dB_voice_dir)
    createpath(snr_based_dir)
    createpath(noisy_save_prediction)
    createpath(path_save_noisy_image)
    createpath(path_save_voice_image)

    list_noise_files = []
    list_voice_files = []
    list_noise_snr_files = []

    print("Loading...")
    for root, dirs, files in os.walk(noise_dir_prediction):
        if len(files) == 0:
            continue

        for f in files:
            files = os.path.join(root, f)
            list_noise_files.append(files)

    for root, dirs, files in os.walk(voice_dir_prediction):
        if len(files) == 0:
            continue

        for f in files:
            files = os.path.join(root, f)
            list_voice_files.append(files)

    target_dBFS = -30.0
    list_dB_noise_file = scale_dB(list_noise_files, dB_noise_dir, target_dBFS)
    list_dB_voice_file = scale_dB(list_voice_files, dB_voice_dir, target_dBFS)

    print("Setting snr...")
    #SNR = [15,10,0,-10,-20]
    SNR = [15, 10, 5, 0, -5]
    for snr in SNR:
        createpath(splitted_noisy_dir+str(snr)+'/')
        createpath(splitted_voice_dir+str(snr)+'/')
        snr_base_noise_file = set_snr(
            list_dB_voice_file, list_dB_noise_file, snr, snr_based_dir, sample_rate)
        list_noise_snr_files.append(snr_base_noise_file)

    loaded_model = load_model(weights_path)
    # TFLITE
    # loaded_model = tf.lite.Interpreter(weights_path)
    # loaded_model = interpreter.get_signature_runner()
    for snr_noise_data in list_noise_snr_files:

        snr = snr_noise_data.split('_')[3].split('/')[1]
        print('snr:', snr)

        noise = audio_files_to_numpy(snr_noise_data, sample_rate)

        voice = audio_files_add_to_numpy(list_dB_voice_file, sample_rate)

        print("Blending...")
        prod_voice, prod_noise, prod_noisy_voice = blend_noise_voice(
            voice, noise, nb_samples, frame_length)

        sf.write(noisy_save_prediction + 'noisy.wav',
                 prod_noise, sample_rate, 'PCM_24')
        sf.write(noisy_save_prediction + 'voice.wav',
                 prod_voice, sample_rate, 'PCM_24')
        sf.write(noisy_save_prediction + 'noisy_voice_long.wav',
                 prod_noisy_voice, sample_rate, 'PCM_24')

        cropped_list_noisy = []
        cropped_list_voice = []

        print("Cropping raw data...")
        cropped_list_noisy.extend(split_into_one_second(
            prod_noisy_voice, splitted_noisy_dir, sample_rate, snr, False))
        cropped_list_voice.extend(split_into_one_second(
            prod_voice, splitted_voice_dir, sample_rate, snr, False))

        dim_square_spec = int(n_fft / 2) + 1

        cropped_array_noisy = np.array(cropped_list_noisy)
        cropped_array_voice = np.array(cropped_list_voice)
        # sf.write(dir_save_prediction + 'Clean.wav',
        #          cropped_array_voice, sample_rate, 'PCM_24')

        print("cropped_array_noisy:", cropped_array_noisy.shape)
        print("cropped_array_voice:", cropped_array_voice.shape)
        fix_length = cropped_array_noisy.shape[1]

        m_amp_db_voice,  m_pha_voice = numpy_audio_to_matrix_spectrogram(
            cropped_array_voice, dim_square_spec, n_fft, hop_length_fft, path_save_voice_image)

        m_amp_db_noisy_voice,  m_pha_noisy_voice = numpy_audio_to_matrix_spectrogram(
            cropped_array_noisy, dim_square_spec, n_fft, hop_length_fft, path_save_noisy_image)

        X_in = scaled_in(m_amp_db_noisy_voice)
        print("X_in:", X_in.shape)
        X_pred = loaded_model.predict(X_in)
        print("X_pred:", X_pred.shape)

        inv_sca_X_pred = inv_scaled_ou(X_pred)
        print("inv_sca_X_pred:", inv_sca_X_pred.shape)
        X_denoise = m_amp_db_noisy_voice - inv_sca_X_pred
        print("X_denoise:", X_denoise.shape)

        audio_denoise_recons = matrix_spectrogram_to_numpy_audio(
            X_denoise, m_pha_noisy_voice, frame_length, hop_length_fft, fix_length)
        nb_samples = audio_denoise_recons.shape[0]

        #denoise_long = audio_denoise_recons.reshape(1, nb_samples * frame_length)*10
        sf.write(dir_save_prediction + snr+'_' + audio_output_prediction,
                 audio_denoise_recons, sample_rate, 'PCM_24')

        output_audio_list = []
        output_audio_list.append(
            dir_save_prediction + snr + '_' + audio_output_prediction)
        scale_dB(output_audio_list, dir_save_prediction, target_dBFS)
        print("\n")


if __name__ == '__main__':

    weights_path = './Train/weights/model_DDAE.h5'
    # weights_path = './Train/weights/model.tflite'
    name_model = 'model_gru.h5'
    dB_noise_dir = './demo_data/test/scaled_dB/noise/'
    dB_voice_dir = './demo_data/test/scaled_dB/voice/'
    noise_dir_prediction = './demo_data/test/NOISE/'
    voice_dir_prediction = './demo_data/test/TIMIT/'
    snr_based_dir = './demo_data/save_predictions/Noise_SNR/'
    splitted_noisy_dir = './demo_data/save_predictions/splitted/noisy/'
    splitted_voice_dir = './demo_data/save_predictions/splitted/voice/'
    dir_save_prediction = './demo_data/save_predictions/'
    noisy_save_prediction = './demo_data/save_predictions/NOISY/'
    path_save_noisy_image = './demo_data/save_predictions/image_noisy/'
    path_save_voice_image = './demo_data/save_predictions/image_voice/'
    audio_output_prediction = 'denoise_t2.wav'
    sample_rate = 16000
    min_duration = 1.0
    frame_length = 18000
    hop_length_frame = 250
    nb_samples = 8000
    n_fft = 511
    hop_length_fft = 313

    prediction(weights_path, name_model, dB_noise_dir, dB_voice_dir, snr_based_dir, splitted_noisy_dir, splitted_voice_dir, noise_dir_prediction, voice_dir_prediction, dir_save_prediction, noisy_save_prediction,
               path_save_noisy_image, path_save_voice_image, audio_output_prediction, sample_rate, min_duration, frame_length, hop_length_frame, nb_samples, n_fft, hop_length_fft)
