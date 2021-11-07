import librosa
import numpy as np
import os
from pydub import AudioSegment
import scipy
from matplotlib import pyplot as plt
from pydub import AudioSegment
import soundfile as sf
import config_params


def createpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


def scale_dB(list_audio_files, dB_audio_dir, purepath=False):
    '''
    Description:
        1. Load audio files by parsing the `list_audio_files`
        2. Apply gain to each audio file
        3. Save these scaled audio files
        4. Append new file path to `list_dB_sound`
        5. Return the `list_dB_sound`
    Args:
        list_audio_files: a list contains file paths
        dB_audio_dir: path to save the scaled audio

    Returns:
        list_dB_sound: a list contains paths of scaled files
    '''

    list_dB_sound = []

    for audio_file in list_audio_files:
        sound = AudioSegment.from_file(audio_file, format="wav")
        loudness = sound.dBFS
        change_in_dBFS = config_params.TARGET_dBFS - loudness

        scaled_audio = sound.apply_gain(change_in_dBFS)
        list_dB_sound.append(dB_audio_dir+audio_file.split('/')[-1])

        if (not purepath):
            scaled_audio.export(
                dB_audio_dir+audio_file.split('/')[-1], format="wav")
        else:
            scaled_audio.export(
                dB_audio_dir + "/" + audio_file.split('/')[-1], format="wav")

    return list_dB_sound


def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame):

    sequence_sample_length = sound_data.shape[0]

    sound_data_list = [sound_data[start:start + frame_length] for start in range(
        0, sequence_sample_length - frame_length + 1, hop_length_frame)]  # get sliding windows
    sound_data_array = np.vstack(sound_data_list)

    return sound_data_array


def audio_files_to_numpy(audio_files):
    list_sound = []

    signal, sr = librosa.load(audio_files, sr=None)

    # Only resmaple if the sample rate of file is not as specified
    if sr != config_params.SAMPLE_RATE:
        signal = librosa.resample(
            signal, orig_sr=sr, target_sr=config_params.SAMPLE_RATE, res_type='polyphase')

    list_sound.extend(signal)
    array_sound = np.array(list_sound)
    return array_sound


def set_snr(list_voice_files, list_noise_files, snr, snr_based_dir):

    for noise_file in list_noise_files:

        noise, sr_noise = librosa.load(noise_file, sr=None)
        voice, sr_voice = librosa.load(list_voice_files[0], sr=None)

        # Only resmaple if the sample rate of file is not as specified
        if sr_noise != config_params.SAMPLE_RATE:
            noise = librosa.resample(
                noise, orig_sr=sr_noise, target_sr=config_params.SAMPLE_RATE, res_type='polyphase')

        # Only resmaple if the sample rate of file is not as specified
        if sr_voice != config_params.SAMPLE_RATE:
            voice = librosa.resample(
                voice, orig_sr=sr_voice, target_sr=config_params.SAMPLE_RATE, res_type='polyphase')

        if snr == np.inf:
            scaler, prescaler = 0, 1
        elif snr == -np.inf:
            scaler, prescaler = 1, 0
        else:
            power1 = np.sum(voice**2)
            power2 = np.sum(noise**2)
            scaler = np.sqrt(power1 / (power2 * 10.**(snr/10.)))
            prescaler = 1

        snr_base_noise_file = snr_based_dir+str(snr)+'_noisy'+'.wav'
        sf.write(snr_base_noise_file, scaler * noise,
                 config_params.SAMPLE_RATE, 'PCM_24')

    return snr_base_noise_file


def noise_files_to_numpy(snr_noise_data, frame_length, hop_length_frame, min_duration):

    list_sound_array = []

    signal, sr = librosa.load(snr_noise_data, sr=None)

    # Only resmaple if the sample rate of file is not as specified
    if sr != config_params.SAMPLE_RATE:
        signal = librosa.resample(
            signal, orig_sr=sr, target_sr=config_params.SAMPLE_RATE, res_type='polyphase')

    total_duration = librosa.get_duration(
        y=signal, sr=config_params.SAMPLE_RATE)

    if (total_duration >= min_duration):
        list_sound_array.append(audio_to_audio_frame_stack(
            signal, frame_length, hop_length_frame))
    else:
        print(
            "The following file {os.path.join(audio_dir,file)} is below the min duration")

    return np.vstack(list_sound_array)


def voice_files_to_numpy(list_audio_files, sample_rate, frame_length, hop_length_frame, min_duration):

    list_sound_array = []

    for file in list_audio_files:
        # Open the audio file
        signal, sr = librosa.load(file, sr=None)

        # Only resmaple if the sample rate of file is not as specified
        if sr != config_params.SAMPLE_RATE:
            signal = librosa.resample(
                signal, orig_sr=sr, target_sr=config_params.SAMPLE_RATE, res_type='polyphase')

        total_duration = librosa.get_duration(
            y=signal, sr=config_params.SAMPLE_RATE)

        if (total_duration >= min_duration):
            list_sound_array.append(audio_to_audio_frame_stack(
                signal, frame_length, hop_length_frame))
        else:
            print(
                "The following file {os.path.join(audio_dir,file)} is below the min duration")

    return np.vstack(list_sound_array)


def mix_noise_randomly(voice, noise, nb_samples, frame_length):

    prod_voice = np.zeros((nb_samples, frame_length))
    prod_noise = np.zeros((nb_samples, frame_length))
    prod_noisy = np.zeros((nb_samples, frame_length))

    for i in range(nb_samples):
        id_voice = np.random.randint(0, voice.shape[0])
        id_noise = np.random.randint(0, noise.shape[0])
        prod_voice[i, :] = voice[id_voice, :]
        prod_noise[i, :] = noise[id_noise, :]
        prod_noisy[i, :] = prod_voice[i, :] + prod_noise[i, :]

    return prod_voice, prod_noise, prod_noisy


def split_into_one_second(sound_data, save_dir, snr, category):
    # get_duration: get the length of the secs
    total_duration = librosa.get_duration(
        y=sound_data, sr=config_params.SAMPLE_RATE)
    splitted_audio_list = []
    save_corpped_list = []
    count = 0

    if(category):
        for time in range(0, int(total_duration)*16000, config_params.SLICE_LENGTH):
            count += 1
            start_time = time
            end_time = time + config_params.SLICE_LENGTH
            # print(total_duration, start_time, end_time)
            SplittedAudio = sound_data[start_time:end_time]
            if(int(SplittedAudio.shape[0]) < config_params.SLICE_LENGTH):
                continue
            splitted_audio_list.append(SplittedAudio)
            # print(save_dir, snr, category)
            sf.write(save_dir + str(snr) + '/' + str(category) + '_' +
                     str(count)+'.wav', SplittedAudio, config_params.SAMPLE_RATE, 'PCM_24')
    else:
        for time in range(0, int(total_duration)*16000, config_params.SLICE_LENGTH):
            count += 1
            start_time = time
            end_time = time + config_params.SLICE_LENGTH
            SplittedAudio = sound_data[start_time:end_time]
            if(int(SplittedAudio.shape[0]) < config_params.SLICE_LENGTH):
                continue
            splitted_audio_list.append(SplittedAudio)
            sf.write(save_dir + str(snr) + '/' + str(count) + '.wav',
                     SplittedAudio, config_params.SAMPLE_RATE, 'PCM_24')
            save_corpped_list.extend(SplittedAudio)

        sf.write(save_dir + str(snr) + '/' + 'Cropped.wav',
                 np.vstack(save_corpped_list), config_params.SAMPLE_RATE, 'PCM_24')

    return splitted_audio_list


def audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, audio, i, path_save_image):

    stftaudio = librosa.stft(
        audio, n_fft=n_fft, hop_length=hop_length_fft, window=scipy.signal.hamming)

    stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)

    stftaudio_magnitude = stftaudio_magnitude[:, 1:-2]
    stftaudio_phase = stftaudio_phase[:, 1:-2]

    stftaudio_magnitude_db = librosa.amplitude_to_db(
        stftaudio_magnitude, ref=np.max)

    # print(stftaudio_magnitude)
    # print(np.abs(stftaudio))

    plt.imsave(os.path.join(path_save_image, str(i) + ".png"),
               stftaudio_magnitude_db, cmap='jet')

    # print(stftaudio_magnitude_db.shape, stftaudio_phase.shape)
    # print(stftaudio_magnitude_db.dtype, stftaudio_phase.dtype)

    return stftaudio_magnitude_db, stftaudio_phase


def numpy_audio_to_matrix_spectrogram(numpy_audio, path_save_image):
    '''
    Args:
        numpy_audio: the numpy array of audio
        path_save_image: str, path to save the spectrogram

    Returns:
        mag_db: magnitude in dB
        mag_phase: phase of magnitude
    '''
    nb_audio = numpy_audio.shape[0]

    mag_tmp, _ = audio_to_magnitude_db_and_phase(
        config_params.N_FFT, config_params.HOP_LENGTH_FFT, numpy_audio[0], 0, path_save_image)

    mag_shape = mag_tmp.shape

    mag_db = np.zeros((nb_audio, mag_shape[0], mag_shape[1]))
    mag_phase = np.zeros(
        (nb_audio, mag_shape[0], mag_shape[1]), dtype=complex)

    for i in range(nb_audio):
        mag_db[i, :, :], mag_phase[i, :, :] = audio_to_magnitude_db_and_phase(
            config_params.N_FFT, config_params.HOP_LENGTH_FFT, numpy_audio[i], i, path_save_image)

    return mag_db, mag_phase


def scaled_in(matrix_spec):
    "global scaling apply to noisy voice spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec + 46)/50
    return matrix_spec


def scaled_ou(matrix_spec):
    "global scaling apply to noise models spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec - 6)/82
    return matrix_spec


def audio_files_add_to_numpy(list_audio_files):
    list_sound = []

    for file in list_audio_files:
        signal, sr = librosa.load(file, sr=None)

        # Only resmaple if the sample rate of file is not as specified
        if sr != config_params.SAMPLE_RATE:
            signal = librosa.resample(
                signal, orig_sr=sr, target_sr=config_params.SAMPLE_RATE, res_type='polyphase')

        list_sound.extend(signal)

    array_sound = np.array(list_sound)

    return array_sound


def mix_noise_voice(voice, noise):

    # If the length of voice is longer than the noise
    # use the length of the noise as the length of the produced noisy voice
    if (voice.shape[0] >= noise.shape[0]):
        prod_noisy = np.zeros((noise.shape[0]))
        prod_noise = np.zeros((noise.shape[0]))
        prod_voice = np.zeros((noise.shape[0]))

        for i in range(noise.shape[0]):
            prod_noise[i, ] = noise[i, ]
            prod_voice[i, ] = voice[i, ]
            prod_noisy[i, ] = prod_voice[i, ] + prod_noise[i, ]
    else:
        prod_noisy = np.zeros((voice.shape[0]))
        prod_noise = np.zeros((voice.shape[0]))
        prod_voice = np.zeros((voice.shape[0]))

        for i in range(voice.shape[0]):
            prod_noise[i, ] = noise[i, ]
            prod_voice[i, ] = voice[i, ]
            prod_noisy[i, ] = prod_voice[i, ] + prod_noise[i, ]

    return prod_voice, prod_noise, prod_noisy


def inv_scaled_ou(matrix_spec):

    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec


def magnitude_db_and_phase_to_audio(hop_length_fft, stftaudio_magnitude_db, stftaudio_phase):

    stftaudio_magnitude_rev = librosa.db_to_amplitude(
        stftaudio_magnitude_db, ref=1.0)

    audio_reverse_stft = stftaudio_magnitude_rev * stftaudio_phase
    audio_reconstruct = librosa.core.istft(
        audio_reverse_stft, hop_length=hop_length_fft, window=scipy.signal.hamming, center=True)

    return audio_reconstruct


def matrix_spectrogram_to_numpy_audio(mag_db, mag_phase, fix_length, path_save_image):

    list_audio = []

    nb_spec = mag_db.shape[0]

    for i in range(nb_spec):

        plt.imsave(os.path.join(path_save_image, str(i) + ".png"),
                   mag_db[i], cmap='jet')

        audio_reconstruct = magnitude_db_and_phase_to_audio(
            config_params.HOP_LENGTH_FFT, mag_db[i], mag_phase[i])

        audio_reconstruct = librosa.util.fix_length(
            audio_reconstruct, fix_length)

        list_audio.extend(audio_reconstruct)

    return np.vstack(list_audio)
