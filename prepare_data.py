import os
import librosa
from data_tools import scale_dB, set_snr, audio_files_add_to_numpy, audio_files_to_numpy, blend_noise_voice, blend_noise_randomly
from data_tools import split_into_one_second, numpy_audio_to_matrix_spectrogram, scaled_in, scaled_ou
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
import h5py
import soundfile as sf
import config_params


def save_h5(h5_file, input_data, target):
    shape_list = list(input_data.shape)
    # Check if the dict doesn't contains the target
    if not h5_file.__contains__(target):
        shape_list[0] = None
        dataset = h5_file.create_dataset(
            target, data=input_data, maxshape=tuple(shape_list), chunks=True)
        # print(dataset)
        return
    else:
        dataset = h5_file[target]
        # print(dataset)
    len_old = dataset.shape[0]
    len_new = len_old + input_data.shape[0]
    shape_list[0] = len_new
    dataset.resize(tuple(shape_list))
    # Append new data to dataset
    dataset[len_old:len_new] = input_data
    print(dataset)


def append_value_to_dict(dict_obj, key, value):
    # Check if key exist in dict For not
    if key in dict_obj:
        # Key exist in dict.
        # Check if type of value of key is list or not
        if not isinstance(dict_obj[key], list):
            # If type is not list then make it list
            dict_obj[key] = [dict_obj[key]]
        # Append the value in list
        dict_obj[key].append(value)
    else:
        # As key is not in dict,
        # so, add key-value pair
        dict_obj[key] = value


def createpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_data():

    createpath(config_params.PATH_DIR_TRAIN_SNR_BASED)
    createpath(config_params.PATH_DIR_TRAIN_NOISE_dB)
    createpath(config_params.PATH_DIR_TRAIN_VOICE_dB)
    createpath(config_params.PATH_DIR_TRAIN_BLENDED_SOUND)
    createpath(config_params.PATH_DIR_TRAIN_TIME_SERIES)
    createpath(config_params.PATH_DIR_TRAIN_DATASET)
    createpath(config_params.PATH_DIR_TRAIN_IMAGE_NOISY)
    createpath(config_params.PATH_DIR_TRAIN_IMAGE_VOICE)

    # list to save file names
    list_noise_files = []
    list_voice_files = []
    list_dB_noise_file = []
    list_dB_voice_file = []
    list_noise_snr_files = []
    list_voice_dict = {}

    print("Loading...")
    # Load noise file paths
    for root, dirs, files in os.walk(config_params.PATH_DIR_NOISE_CLASS):
        print(config_params.PATH_DIR_NOISE_CLASS)
        if len(files) == 0:
            continue

        for f in files:
            files = os.path.join(root, f)
            list_noise_files.append(files)

    # Load voice filenames
    # list_voice_dict == {"TRAIN": [voice file paths]}
    for root, dirs, files in os.walk(config_params.PATH_DIR_VOICE_SOURCE):
        if len(files) == 0:
            continue

        root_dir = root.split('/')[-2]
        for f in files:
            files = os.path.join(root, f)

            append_value_to_dict(list_voice_dict, root_dir, files)

    # print(list_voice_dict)

    # It only has a key "TRAIN"
    for keys, values in list_voice_dict.items():
        if len(values) == 0:
            continue

        # voice_path: ./Train/Time_serie/TRAIN_voice.wav
        voice_path = config_params.PATH_DIR_TRAIN_TIME_SERIES+str(keys)+'_voice'+'.wav'
        # list_voice_files: ['./Train/Time_serie/TRAIN_voice.wav']
        list_voice_files.append(voice_path)
        # Use librosa.load() to load the voice files in the list
        voice = audio_files_add_to_numpy(values, config_params.SAMPLE_RATE)
        # Save as a single file
        sf.write(voice_path, voice, config_params.SAMPLE_RATE, 'PCM_24')

    # Remove unused dict
    list_voice_dict.clear()

    target_dBFS = -30.0
    # Normalize the audio files
    list_dB_noise_file = scale_dB(list_noise_files, config_params.PATH_DIR_TRAIN_NOISE_dB, target_dBFS)
    list_dB_voice_file = scale_dB(list_voice_files, config_params.PATH_DIR_TRAIN_VOICE_dB, target_dBFS)

    # Remove unused list
    list_noise_files.clear()
    list_voice_files.clear()

    print("Setting snr...")
    # SNR = [20, 10, 0, -10, -20]
    SNR = [15, 10, 5, 0, -5, -10]
    # Here we only scale the noise
    for snr in SNR:
        createpath(config_params.PATH_DIR_TRAIN_SPLIT_NOISY+str(snr)+'/')
        createpath(config_params.PATH_DIR_TRAIN_SPLIT_VOICE+str(snr)+'/')
        # config_params.PATH_DIR_TRAIN_SNR_BASED: the path to save the mixed audio files
        # snr_base_noise_file: the path of the output mixed audio files
        snr_base_noise_file = set_snr(
            list_dB_voice_file, list_dB_noise_file, snr, config_params.PATH_DIR_TRAIN_SNR_BASED, config_params.SAMPLE_RATE)
        # The list to save these paths(every snr_base_noise_file)
        list_noise_snr_files.append(snr_base_noise_file)

    for snr_noise_data in list_noise_snr_files:

        cropped_list_noisy = []
        cropped_list_voice = []
        snr = snr_noise_data.split('/')[-1].split('_')[0]
        print('snr:', snr)

        for voice_data in list_dB_voice_file:
            list_noise = []
            # Load files as numpy array
            noise = audio_files_to_numpy(snr_noise_data, config_params.SAMPLE_RATE)
            voice = audio_files_to_numpy(voice_data, config_params.SAMPLE_RATE)

            repeat_time = len(voice)//len(noise)

            for time in range(repeat_time):
                list_noise.extend(noise)

            array_noise = np.array(list_noise)

            print("Creating blended audio...")
            # NB_SAMPLES and FRAME_SIZE aren't used in blend_noise_voice()
            prod_voice, prod_noise, prod_noisy_voice = blend_noise_voice(
                voice, array_noise)

            # category: voice
            category = voice_data.split('/')[-2]

            sf.write(config_params.PATH_DIR_TRAIN_BLENDED_SOUND + str(category) + '_' + str(snr) +
                     '_noisy.wav', prod_noise, config_params.SAMPLE_RATE, 'PCM_24')
            sf.write(config_params.PATH_DIR_TRAIN_BLENDED_SOUND + str(category) + '_' + str(snr) +
                     '_voice.wav', prod_voice, config_params.SAMPLE_RATE, 'PCM_24')
            sf.write(config_params.PATH_DIR_TRAIN_BLENDED_SOUND + str(category) + '_' + str(snr) +
                     '_noisy_voice_long.wav', prod_noisy_voice, config_params.SAMPLE_RATE, 'PCM_24')

            print("Splitting raw data...")
            cropped_list_noisy.extend(split_into_one_second(
                prod_noisy_voice, config_params.PATH_DIR_TRAIN_SPLIT_NOISY, config_params.SAMPLE_RATE, snr, category))
            cropped_list_voice.extend(split_into_one_second(
                prod_voice, config_params.PATH_DIR_TRAIN_SPLIT_VOICE, config_params.SAMPLE_RATE, snr, category))

            # prod_noisy_voice:  [0.00471174 0.00608557 0.00568011 ... 0.01218804 0.01283987 0.        ]
            # cropped_list_noisy:  [array([ 0.00471174,  0.00608557,  0.00568011, ...,  0.00347486,

        print("Creating Amplitude and phase...")
        # int(511/2) + 1 = 256
        dim_square_spec = int(config_params.N_FFT / 2) + 1,

        cropped_array_noisy = np.array(cropped_list_noisy)
        cropped_array_voice = np.array(cropped_list_voice)

        # print('test file saved')
        # sf.write('./test.wav', cropped_array_noisy, SAMPLE_RATE, 'PCM_24')

        print("cropped_array_noisy:",
              cropped_array_noisy.shape, type(cropped_array_noisy))
        print("cropped_array_voice:", cropped_array_voice.shape)

        # (numpy_audio, 256, 511, 313, dir to save image)
        m_amp_db_voice,  m_pha_voice = numpy_audio_to_matrix_spectrogram(
            cropped_array_voice, dim_square_spec, config_params.N_FFT, config_params.HOP_LENGTH_FFT, config_params.PATH_DIR_TRAIN_IMAGE_VOICE)
        # m_amp_db_noise,  m_pha_noise = numpy_audio_to_matrix_spectrogram(
        #     prod_noise, dim_square_spec, N_FFT, HOP_LENGTH_FFT)
        m_amp_db_noisy_voice,  m_pha_noisy_voice = numpy_audio_to_matrix_spectrogram(
            cropped_array_noisy, dim_square_spec, config_params.N_FFT, config_params.HOP_LENGTH_FFT, config_params.PATH_DIR_TRAIN_IMAGE_NOISY)

        m_amp_db_voice_scaled = m_amp_db_noisy_voice - m_amp_db_voice

        # reshape(-1, 1): reshape unknown row matrix to 1 column matrix
        print("X_in before scaled:\n", stats.describe(
            m_amp_db_noisy_voice.reshape(-1, 1)))
        print("X_ou before scaled:\n", stats.describe(
            m_amp_db_voice_scaled.reshape(-1, 1)))

        m_amp_db_noisy_voice_scaled = scaled_in(m_amp_db_noisy_voice)
        m_amp_db_voice_scaled = scaled_ou(m_amp_db_voice_scaled)

        print("X_in after scaled:\n", stats.describe(
            m_amp_db_noisy_voice_scaled.reshape(-1, 1)))
        print("X_ou after scaled:\n", stats.describe(
            m_amp_db_voice_scaled.reshape(-1, 1)))

        X_train, X_test, y_train, y_test = train_test_split(
            m_amp_db_noisy_voice_scaled, m_amp_db_voice_scaled, test_size=0.10, random_state=44)

        # 'a', Read/write if exists, create otherwise
        save_h5(h5py.File(f'{config_params.PATH_DIR_TRAIN_DATASET}/amp_db.h5', 'a'),
                input_data=X_train, target='trainnoise')
        save_h5(h5py.File(f'{config_params.PATH_DIR_TRAIN_DATASET}/amp_db.h5', 'a'),
                input_data=y_train, target='trainclean')
        save_h5(h5py.File(f'{config_params.PATH_DIR_TRAIN_DATASET}/amp_db.h5', 'a'),
                input_data=X_test, target='valnoise')
        save_h5(h5py.File(f'{config_params.PATH_DIR_TRAIN_DATASET}/amp_db.h5', 'a'),
                input_data=y_test, target='valclean')

        save_h5(h5py.File(f'{config_params.PATH_DIR_TRAIN_DATASET}/pha_db.h5', 'a'),
                input_data=m_pha_voice, target='trainclean')
        save_h5(h5py.File(f'{config_params.PATH_DIR_TRAIN_DATASET}/pha_db.h5', 'a'),
                input_data=m_pha_noisy_voice, target='trainnoise')

        print("\n")


if __name__ == '__main__':
    create_data()
