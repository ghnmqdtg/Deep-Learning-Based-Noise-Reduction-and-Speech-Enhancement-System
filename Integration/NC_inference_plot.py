import matplotlib.pyplot as plt
import vggish_input as vi
import numpy as np
import soundfile as sf
import vggish_params as params
import tensorflow as tf
import json
import resampy


def inference(model, sound, sr=params.SAMPLE_RATE, result_1D=True, plot=False, ):
    '''
      Input wav file path or audio array and make inference

      Parameters
      ----------
      model: NC keras model

      sound: wav file path or audio array

      sr: sample rate, applied when sound is audio array

      result_1D: If True, return sum of scores of different time slice. If False, return 2D array, [time slice][class].

      plot: whether to plot. Plotting spectrogram is still under construction

      Return
      ------
      .shape = param.NUM_CLASS if reult_1D == True
      .shape = (Depends_on_audio_length, param.NUM_CLASS)
    '''
    if type(sound) is str:
        # spectrogram = vi.wavfile_to_examples(sound)
        sound, sr = sf.read(sound, dtype='float32')
        mel_spectrogram, stft_mag, stft_phase = vi.waveform_to_examples(sound, sr)
    else:
        mel_spectrogram, stft_mag, stft_phase = vi.waveform_to_examples(
            sound, sr)
        mel_spectrogram = np.array(mel_spectrogram)

    model_input_shape = (
        mel_spectrogram.shape[0], mel_spectrogram.shape[1], mel_spectrogram.shape[2], 1)
    scores = model(mel_spectrogram.reshape(model_input_shape))
    scores = np.array(scores)

    # Visualize the results.
    if plot:
        # Get class names
        with open(params.PATH_NOISE_LIST) as json_file:
            dataset_paths = json.load(json_file)
            class_names = list(dataset_paths['train'].keys())

        plt.figure(figsize=(10, 8))

        # Plot the waveform.
        plt.subplot(3, 1, 1)
        plt.plot(sound)
        plt.xlim([0, len(sound)])

        # Plot the log-mel mel_spectrogram
        # mel_spectrogram_2D = mel_spectrogram.reshape([])
        # plt.subplot(3, 1, 2)
        # plt.imshow(mel_spectrogram.T, aspect='auto', interpolation='nearest', origin='lower')

        # Plot scores
        # Plot and label the model output scores for the top-scoring classes.
        plt.subplot(3, 1, 3)
        plt.imshow(scores.T, aspect='auto', cmap='binary')
        # Compensate for the patch_window_seconds (0.96s) context window to align with mel_spectrogram.
        patch_padding = (params.EXAMPLE_WINDOW_SECONDS / 2) / \
            params.EXAMPLE_HOP_SECONDS
        plt.xlim([-patch_padding, scores.shape[0] + patch_padding])
        # Label the top_N classes.
        plt.yticks(range(len(class_names)), class_names)
        _ = plt.ylim(-0.5 + np.array([len(class_names), 0]))

    # Make inference result as 1D array
    if result_1D:
        scores = np.sum(scores, axis=0)

    return scores, stft_mag, stft_phase


if __name__ == '__main__':

    # nc_model = tf.keras.models.load_model(
    #     'generated\weights\model_0806-02_52c5aec_trainAcc91_evalAcc85.h5')

    nc_model = tf.keras.models.load_model(
        './model_0928_3_cats_equal_samples.h5')

    sound, sr = sf.read(
        'test_wav/noisy_voice_long.wav', dtype='float32')

    result, stft_mag, stft_phase = inference(nc_model, sound, sr)
    print(result)
    index = np.argmax(result)
    print(index)

    # print(stft_mag, stft_mag.shape)
    # print(stft_phase, stft_phase.shape)
    print(stft_mag.shape)
    print(stft_phase.shape)
