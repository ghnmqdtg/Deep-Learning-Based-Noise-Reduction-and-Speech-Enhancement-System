import os
import librosa
from data_tools import scale_dB, set_snr, audio_files_add_to_numpy, audio_files_to_numpy, blend_noise_voice, blend_noise_randomly
from data_tools import split_into_one_second, numpy_audio_to_matrix_spectrogram, scaled_in, scaled_ou
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
import h5py
import soundfile as sf


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
    len_new = len_old+input_data.shape[0]
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


def create_data(noise_dir, voice_dir, snr_based_dir, dB_noise_dir, dB_voice_dir, path_save_time_serie, path_save_sound, path_save_spectrogram, splitted_noisy_dir, splitted_voice_dir,
                path_save_noisy_image, path_save_voice_image, sample_rate, min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples, n_fft, hop_length_fft):

    createpath(snr_based_dir)
    createpath(dB_noise_dir)
    createpath(dB_voice_dir)
    createpath(path_save_sound)
    createpath(path_save_time_serie)
    createpath(path_save_spectrogram)
    createpath(path_save_noisy_image)
    createpath(path_save_voice_image)

    # list to save file names
    list_noise_files = []
    list_voice_files = []
    list_dB_noise_file = []
    list_dB_voice_file = []
    list_noise_snr_files = []
    list_voice_dict = {}

    print("Loading...")
    # Load noise file paths
    for root, dirs, files in os.walk(noise_dir):
        if len(files) == 0:
            continue

        for f in files:
            files = os.path.join(root, f)
            list_noise_files.append(files)

    # Load voice filenames
    # list_voice_dict == {"TRAIN": [voice file paths]}
    for root, dirs, files in os.walk(voice_dir):
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
        voice_path = path_save_time_serie+str(keys)+'_voice'+'.wav'
        # list_voice_files: ['./Train/Time_serie/TRAIN_voice.wav']
        list_voice_files.append(voice_path)
        # Use librosa.load() to load the voice files in the list
        voice = audio_files_add_to_numpy(values, sample_rate)
        # Save as a single file
        sf.write(voice_path, voice, sample_rate, 'PCM_24')

    target_dBFS = -30.0
    # Normalize the audio files
    list_dB_noise_file = scale_dB(list_noise_files, dB_noise_dir, target_dBFS)
    list_dB_voice_file = scale_dB(list_voice_files, dB_voice_dir, target_dBFS)

    print("Setting snr...")
    # SNR = [20, 10, 0, -10, -20]
    SNR = [15, 10, 5, 0, -5, -10]
    # Here we only scale the noise
    for snr in SNR:
        createpath(splitted_noisy_dir+str(snr)+'/')
        createpath(splitted_voice_dir+str(snr)+'/')
        # snr_based_dir: the path to save the mixed audio files
        # snr_base_noise_file: the path of the output mixed audio files
        snr_base_noise_file = set_snr(
            list_dB_voice_file, list_dB_noise_file, snr, snr_based_dir, sample_rate)
        # The list to save these paths(every snr_base_noise_file)
        list_noise_snr_files.append(snr_base_noise_file)

    for snr_noise_data in list_noise_snr_files:

        cropped_list_noisy = []
        cropped_list_voice = []
        snr = snr_noise_data.split('_')[1].split('/')[1]
        print('snr:', snr)

        for voice_data in list_dB_voice_file:
            list_noise = []
            # Load files as numpy array
            noise = audio_files_to_numpy(snr_noise_data, sample_rate)
            voice = audio_files_to_numpy(voice_data, sample_rate)

            repeat_time = len(voice)//len(noise)

            for time in range(repeat_time):
                list_noise.extend(noise)

            array_noise = np.array(list_noise)

            print("Creating blended audio...")
            # nb_samples and frame_length aren't used in blend_noise_voice()
            prod_voice, prod_noise, prod_noisy_voice = blend_noise_voice(
                voice, array_noise, nb_samples, frame_length)

            # category: voice
            category = voice_data.split('_')[-2].split('/')[1]

            sf.write(path_save_sound + str(category) + '_' + str(snr) +
                     '_noisy.wav', prod_noise, sample_rate, 'PCM_24')
            sf.write(path_save_sound + str(category) + '_' + str(snr) +
                     '_voice.wav', prod_voice, sample_rate, 'PCM_24')
            sf.write(path_save_sound + str(category) + '_' + str(snr) +
                     '_noisy_voice_long.wav', prod_noisy_voice, sample_rate, 'PCM_24')

            print("Splitting raw data...")
            cropped_list_noisy.extend(split_into_one_second(
                prod_noisy_voice, splitted_noisy_dir, sample_rate, snr, category))
            cropped_list_voice.extend(split_into_one_second(
                prod_voice, splitted_voice_dir, sample_rate, snr, category))

            # prod_noisy_voice:  [0.00471174 0.00608557 0.00568011 ... 0.01218804 0.01283987 0.        ]
            # cropped_list_noisy:  [array([ 0.00471174,  0.00608557,  0.00568011, ...,  0.00347486,

        print("Creating Amplitude and phase...")
        # int(511/2) + 1 = 256
        dim_square_spec = int(n_fft / 2) + 1

        cropped_array_noisy = np.array(cropped_list_noisy)
        cropped_array_voice = np.array(cropped_list_voice)

        # print('test file saved')
        # sf.write('./test.wav', cropped_array_noisy, sample_rate, 'PCM_24')

        print("cropped_array_noisy:",
              cropped_array_noisy.shape, type(cropped_array_noisy))
        print("cropped_array_voice:", cropped_array_voice.shape)

        # (numpy_audio, 256, 511, 313, dir to save image)
        m_amp_db_voice,  m_pha_voice = numpy_audio_to_matrix_spectrogram(
            cropped_array_voice, dim_square_spec, n_fft, hop_length_fft, path_save_voice_image)
        # m_amp_db_noise,  m_pha_noise = numpy_audio_to_matrix_spectrogram(
        #     prod_noise, dim_square_spec, n_fft, hop_length_fft)
        m_amp_db_noisy_voice,  m_pha_noisy_voice = numpy_audio_to_matrix_spectrogram(
            cropped_array_noisy, dim_square_spec, n_fft, hop_length_fft, path_save_noisy_image)

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
        save_h5(h5py.File(path_save_spectrogram + "amp_db.h5", 'a'),
                input_data=X_train, target='trainnoise')
        save_h5(h5py.File(path_save_spectrogram + "amp_db.h5", 'a'),
                input_data=y_train, target='trainclean')
        save_h5(h5py.File(path_save_spectrogram + "amp_db.h5", 'a'),
                input_data=X_test, target='valnoise')
        save_h5(h5py.File(path_save_spectrogram + "amp_db.h5", 'a'),
                input_data=y_test, target='valclean')

        save_h5(h5py.File(path_save_spectrogram + "pha_db.h5", 'a'),
                input_data=m_pha_voice, target='trainclean')
        save_h5(h5py.File(path_save_spectrogram + "pha_db.h5", 'a'),
                input_data=m_pha_noisy_voice, target='trainnoise')

        # np.save(path_save_spectrogram + "voice_amp_db.npy",np.append(voice_amp_db,m_amp_db_voice))
        # np.save(path_save_spectrogram + "noisy_voice_amp_db.npy",np.append(noisy_voice_amp_db,m_amp_db_noisy_voice))

        # np.save(path_save_spectrogram + "voice_pha_db.npy",np.append(voice_pha_db,m_pha_voice))
        # np.save(path_save_spectrogram + "noisy_voice_pha_db.npy",np.append(noisy_voice_pha_db,m_pha_noisy_voice))

        print("\n")

    # np.save(path_save_time_serie + 'voice_timeserie', prod_voice)
    # np.save(path_save_time_serie + 'noise_timeserie', prod_noise)
    # np.save(path_save_time_serie + 'noisy_voice_timeserie', prod_noisy_voice)

    '''
    np.save(path_save_spectrogram + 'voice_amp_db', m_amp_db_voice)
    #np.save(path_save_spectrogram + 'noise_amp_db', m_amp_db_noise)
    np.save(path_save_spectrogram + 'noisy_voice_amp_db', m_amp_db_noisy_voice)

    np.save(path_save_spectrogram + 'voice_pha_db', m_pha_voice)
    #np.save(path_save_spectrogram + 'noise_pha_db', m_pha_noise)
    np.save(path_save_spectrogram + 'noisy_voice_pha_db', m_pha_noisy_voice)
    '''


if __name__ == '__main__':
    noise_dir = './NOISE/Household_Appliance/'
    voice_dir = './TIMIT_CONVERTED_NORMALIZED/TRAIN/'
    snr_based_dir = './Train/Noise_SNR/'
    dB_noise_dir = './Train/scaled_dB/noise/'
    dB_voice_dir = './Train/scaled_dB/voice/'
    path_save_sound = './Train/sound/'
    splitted_noisy_dir = './Train/splitted/noisy/'
    splitted_voice_dir = './Train/splitted/voice/'
    path_save_time_serie = './Train/Time_serie/'
    path_save_spectrogram = './Train/spectrogram/'
    path_save_noisy_image = './Train/noisy_image/'
    path_save_voice_image = './Train/voice_image/'
    sample_rate = 16000
    min_duration = 1.0
    frame_length = 18000
    hop_length_frame_voice = 250
    hop_length_frame_noise = 250
    nb_samples = 50
    n_fft = 511
    hop_length_fft = 313

    create_data(noise_dir, voice_dir, snr_based_dir, dB_noise_dir, dB_voice_dir, path_save_time_serie, path_save_sound, path_save_spectrogram, splitted_noisy_dir, splitted_voice_dir,
                path_save_noisy_image, path_save_voice_image, sample_rate, min_duration, frame_length, hop_length_frame_voice, hop_length_frame_noise, nb_samples, n_fft, hop_length_fft)
