import os
import librosa
import numpy as np
import soundfile as sf


def audio_files_add_to_numpy(list_audio_files):

    list_sound = []

    for file in list_audio_files:
        y, sr = librosa.load(file, sr=None)

        list_sound.extend(y[:80000])

    array_sound = np.array(list_sound)

    return array_sound

def slice_into_one_second(file):

    y, sr = librosa.load(file, sr=None)
    sound = y[:16000]

    return np.array(sound)


if __name__ == '__main__':
    # list_noise_files = []

    # for root, dirs, files in os.walk("./demo_data/Vechicles/test/noise_source"):
    #     if len(files) == 0:
    #         continue

    #     for f in files:
    #         files = os.path.join(root, f)
    #         list_noise_files.append(files.replace("\\", "/"))

    # # Save as a single file
    # sf.write("./demo_data/Vechicles/test/NOISE/vechicles_evaluation.wav", audio_files_add_to_numpy(
    #     list_noise_files), 16000, 'PCM_24')

    one_sec_sound = slice_into_one_second(
        "./demo_data/Vechicles/save_predictions/0_denoise_t2.wav")

    sf.write("./one_sec_Vechicles.wav", one_sec_sound, 16000, 'PCM_24')
