import librosa
import numpy as np
import os
from pydub import AudioSegment
import scipy
from matplotlib import pyplot as plt
from pydub import AudioSegment
import soundfile as sf


def createpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


def scale_dB(list_audio_files, dB_audio_dir, target_dBFS):
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
        target_dBFS: dBFS we want to scale the audio

    Returns:
        list_dB_sound: a list contains paths of scaled files
    '''

    list_dB_sound = []

    for audio_file in list_audio_files:
        sound = AudioSegment.from_file(audio_file, format="wav")
        loudness = sound.dBFS
        change_in_dBFS = target_dBFS - loudness

        scaled_audio = sound.apply_gain(change_in_dBFS)
        list_dB_sound.append(dB_audio_dir+audio_file.split('/')[-1])
        scaled_audio.export(
            dB_audio_dir+audio_file.split('/')[-1], format="wav")

    return list_dB_sound


def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame):

    sequence_sample_length = sound_data.shape[0]

    sound_data_list = [sound_data[start:start + frame_length] for start in range(
        0, sequence_sample_length - frame_length + 1, hop_length_frame)]  # get sliding windows
    sound_data_array = np.vstack(sound_data_list)

    return sound_data_array


def audio_files_to_numpy(audio_files, sample_rate):

    list_sound = []

    y, sr = librosa.load(audio_files, sr=sample_rate)

    list_sound.extend(y)

    array_sound = np.array(list_sound)

    return array_sound


def set_snr(list_voice_files, list_noise_files, snr, snr_based_dir, sample_rate):

    for noise_file in list_noise_files:

        noise, sr_noise = librosa.load(noise_file, sr=sample_rate)
        voice, sr_voice = librosa.load(list_voice_files[0], sr=sample_rate)

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
        sf.write(snr_base_noise_file, scaler * noise, sample_rate, 'PCM_24')

    return snr_base_noise_file


def noise_files_to_numpy(snr_noise_data, sample_rate, frame_length, hop_length_frame, min_duration):

    list_sound_array = []

    y, sr = librosa.load(snr_noise_data, sr=sample_rate)
    total_duration = librosa.get_duration(y=y, sr=sr)

    if (total_duration >= min_duration):
        list_sound_array.append(audio_to_audio_frame_stack(
            y, frame_length, hop_length_frame))
    else:
        print(
            "The following file {os.path.join(audio_dir,file)} is below the min duration")

    return np.vstack(list_sound_array)


def voice_files_to_numpy(list_audio_files, sample_rate, frame_length, hop_length_frame, min_duration):

    list_sound_array = []

    for file in list_audio_files:
        # open the audio file
        y, sr = librosa.load(file, sr=sample_rate)
        total_duration = librosa.get_duration(y=y, sr=sr)

        if (total_duration >= min_duration):
            list_sound_array.append(audio_to_audio_frame_stack(
                y, frame_length, hop_length_frame))
        else:
            print(
                "The following file {os.path.join(audio_dir,file)} is below the min duration")

    return np.vstack(list_sound_array)


def blend_noise_randomly(voice, noise, nb_samples, frame_length):

    prod_voice = np.zeros((nb_samples, frame_length))
    prod_noise = np.zeros((nb_samples, frame_length))
    prod_noisy_voice = np.zeros((nb_samples, frame_length))

    for i in range(nb_samples):
        id_voice = np.random.randint(0, voice.shape[0])
        id_noise = np.random.randint(0, noise.shape[0])
        #level_noise = np.random.uniform(0.2, 0.8)
        prod_voice[i, :] = voice[id_voice, :]
        prod_noise[i, :] = noise[id_noise, :]
        #prod_noise[i, :] = level_noise * noise[id_noise, :]
        prod_noisy_voice[i, :] = prod_voice[i, :] + prod_noise[i, :]

    return prod_voice, prod_noise, prod_noisy_voice


def split_into_one_second(sound_data, save_dir, sample_rate, snr, category):

    # audio, sr = librosa.load(sound_data, sr=sample_rate)
    # get_duration: get the length of the secs
    total_duration = librosa.get_duration(y=sound_data, sr=sample_rate)
    splitted_audio_list = []
    save_corpped_list = []
    count = 0
    if(category):
        for time in range(0, int(total_duration)*16000, 80000):
            count += 1
            start_time = time
            end_time = time + 80000
            # print(total_duration, start_time, end_time)
            SplittedAudio = sound_data[start_time:end_time]
            if(int(SplittedAudio.shape[0]) < 80000):
                continue
            splitted_audio_list.append(SplittedAudio)
            sf.write(save_dir + str(snr) + '/' + str(category) + '_' +
                     str(count)+'.wav', SplittedAudio, sample_rate, 'PCM_24')
    else:
        for time in range(0, int(total_duration)*16000, 80000):
            count += 1
            start_time = time
            end_time = time + 80000
            SplittedAudio = sound_data[start_time:end_time]
            if(int(SplittedAudio.shape[0]) < 80000):
                continue
            splitted_audio_list.append(SplittedAudio)
            sf.write(save_dir + str(snr) + '/' + str(count) + '.wav',
                     SplittedAudio, sample_rate, 'PCM_24')
            save_corpped_list.extend(SplittedAudio)

        sf.write(save_dir + str(snr) + '/' + 'Cropped.wav',
                 np.vstack(save_corpped_list), sample_rate, 'PCM_24')

    return splitted_audio_list


def audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, audio, i, path_save_image):

    stftaudio = librosa.stft(
        audio, n_fft=n_fft, hop_length=hop_length_fft, window=scipy.signal.hamming)
    stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)

    stftaudio_magnitude_db = librosa.amplitude_to_db(
        stftaudio_magnitude, ref=np.max)

    # print(stftaudio_magnitude)
    # print(np.abs(stftaudio))

    plt.imsave(os.path.join(path_save_image, str(i) + ".png"),
               stftaudio_magnitude_db, cmap='jet')

    return stftaudio_magnitude_db, stftaudio_phase


def numpy_audio_to_matrix_spectrogram(numpy_audio, dim_square_spec, n_fft, hop_length_fft, path_save_image):
    '''
    Args:
        numpy_audio: the numpy array of audio
        dim_square_spec: int, the dimension of the square spectrogram
                         default = int(n_fft / 2) + 1 = 256
        n_fft: int, default = 511
        hop_length_fft: int, default = 313
        path_save_image: str, path to save the spectrogram

    Returns:
        m_mag_db: magnitude in dB
        m_phase: phase of magnitude
    '''
    nb_audio = numpy_audio.shape[0]

    m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
    m_phase = np.zeros(
        (nb_audio, dim_square_spec, dim_square_spec), dtype=complex)

    for i in range(nb_audio):
        m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_db_and_phase(
            n_fft, hop_length_fft, numpy_audio[i], i, path_save_image)

    return m_mag_db, m_phase


def scaled_in(matrix_spec):
    "global scaling apply to noisy voice spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec + 46)/50
    return matrix_spec


def scaled_ou(matrix_spec):
    "global scaling apply to noise models spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec - 6)/82
    return matrix_spec


def audio_files_add_to_numpy(list_audio_files, sample_rate):

    list_sound = []

    for file in list_audio_files:
        # y: [-2.4414062e-04 -3.0517578e-05  1.8310547e-04 ...  1.2207031e-04
        # 9.1552734e-05  1.2207031e-04]
        y, sr = librosa.load(file, sr=sample_rate)

        list_sound.extend(y)

    array_sound = np.array(list_sound)

    return array_sound


def blend_noise_voice(voice, noise, nb_samples, frame_length):

    # If the length of voice is longer than the noise
    # use the length of the noise as the length of the produced noisy voice
    if (voice.shape[0] >= noise.shape[0]):
        prod_noisy_voice = np.zeros((noise.shape[0]))
        prod_noise = np.zeros((noise.shape[0]))
        prod_voice = np.zeros((noise.shape[0]))

        for i in range(noise.shape[0]):
            prod_noise[i, ] = noise[i, ]
            prod_voice[i, ] = voice[i, ]
            prod_noisy_voice[i, ] = prod_voice[i, ] + prod_noise[i, ]
    else:
        prod_noisy_voice = np.zeros((voice.shape[0]))
        prod_noise = np.zeros((voice.shape[0]))
        prod_voice = np.zeros((voice.shape[0]))

        for i in range(voice.shape[0]):
            prod_noise[i, ] = noise[i, ]
            prod_voice[i, ] = voice[i, ]
            prod_noisy_voice[i, ] = prod_voice[i, ] + prod_noise[i, ]

    return prod_voice, prod_noise, prod_noisy_voice


def inv_scaled_ou(matrix_spec):

    matrix_spec = matrix_spec * 82 + 6
    return matrix_spec


def magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, stftaudio_magnitude_db, stftaudio_phase):

    stftaudio_magnitude_rev = librosa.db_to_amplitude(
        stftaudio_magnitude_db, ref=1.0)

    audio_reverse_stft = stftaudio_magnitude_rev * stftaudio_phase
    audio_reconstruct = librosa.core.istft(
        audio_reverse_stft, hop_length=hop_length_fft, window=scipy.signal.hamming, center=True)
    #audio_reconstruct = librosa.core.istft(audio_reverse_stft, hop_length=hop_length_fft)

    return audio_reconstruct


def matrix_spectrogram_to_numpy_audio(m_mag_db, m_phase, frame_length, hop_length_fft, fix_length):

    list_audio = []

    nb_spec = m_mag_db.shape[0]

    for i in range(nb_spec):

        audio_reconstruct = magnitude_db_and_phase_to_audio(
            frame_length, hop_length_fft, m_mag_db[i], m_phase[i])

        audio_reconstruct = librosa.util.fix_length(
            audio_reconstruct, fix_length)  # ADD

        list_audio.extend(audio_reconstruct)

    return np.vstack(list_audio)
