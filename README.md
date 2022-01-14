# Deep Learning Based Noise Reduction and Speech Enhancement System

## Introduction
People with hearing loss depend heavily on hearing aid to hear properly. However, traditional hearing aid also amplify disturbing environmental noise. This project implements **two deep learning models, one can classify the type of noise, the other can retain human voice and reduce environmental noise**. This proposal has the advantage of adapting to environmental noise compared to the traditional hearing aid.

In this repo, we only provide codes of data preprocessing, model training and model validation that runs on PC.

## Structure
![](https://i.imgur.com/XduRFpP.png)

## Total works
- Audio to Spectrogram Preprocessing & Inversion Algorithm (STFT & MFCC)
- Dataset Generator: Human Voice & Environmental Noise Mixing
- Noise Classifier (NC) Model
- Noise Reduction Model (DDAE)
- Preprocessing, NC and DDAE Integration on PC
- Model conversion: Implement quantization and convert into tflite model
- Deploy to embedded system & optimize the performance
- Wording & Intro Video Editing

## Datasets
The goal of this project is to recognize the noise in the input audio and clean them out with deep learning techniques. We mix clean voice and noises together as training input, and clean voice as output.

As we all know, the data you use is more important than the model itself. We choose TIMIT as clean voice dataset. And according to [*Sounds perceived as annoying by hearing-aid users in their daily soundscape*](https://www.researchgate.net/publication/260093127_Sounds_perceived_as_annoying_by_hearing-aid_users_in_their_daily_soundscape), we select four categories of noise that respondents mentioned most, which are verbal human sounds, TV/radio, vehicles and household appliances as the goal that we want to clean out from the input audio. We sort out the labels from Audioset to build our target noise categories.

## Models
### Noise Classifier, NC
![](https://i.imgur.com/LOnqGXX.png)

This section tells setup of Noise Classifier. We use another [repo](https://github.com/qqq89513/NOISE_CLASSIFIER) for it, so please `cd` another folder before continues.
```
git clone https://github.com/qqq89513/NOISE_CLASSIFIER
```
The repo itself does not contain dataset and the trained weights. The weights will be published as soon as the training stage is finalized.
Here are some public sources that we used for Noise Classifier:
  - Dataset: TIMIT mixed with AudioSet
  - Preprocessing reference: [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)
File desciption:
  - `utils/data_paser.py`: This script generates the list of dataset file paths as json. You can execute the script directly:
      ```
      python ./utils/data_paser.py
      ```
  - `requirement.txt`: The dependencies of this repo, will be available soon.
  - `vggish_params.py`: Hyperparameters.
  - `vggish_input.py`: Converter from audio waveform into input examples.
  - `vggish_mel_features.py`: Audio feature extraction helpers. This is a pure nummpy implementation, which can be referenced for embedded implementation.
  - `NC_training.py`
    1. Loads dataset from the disk
    2. Preprocesses via `vggish_input.py`
    3. Build the classifier model
    4. Train the model
    5. Inference

### Deep Denoising Autoencoder, DDAE
![](https://i.imgur.com/lXgRIEV.png)

DDAE is one of the derivatives of the autoencoder. Autoencoders are neural networks trained in an unsupervised way to attempt to copy inputs to outputs. And denoise autoencoder(DAE) is trained to reduce the noise from the input. In this project, we add multiple layers to the DAE and makes it DDAE.

After preprocessing the input audio to the spectrogram, we feed it to the trained DDAE model, and the model will reduce the noise from the input, then you will get a clean spectrogram. You will need to do the inverse DSP to convert is to a listenable wav file or output it instantly.

# Results
### Noise Classifier, NC
![](https://i.imgur.com/ikwvxMc.png)

|        Class         |               Results                |
|:--------------------:|:------------------------------------:|
|     Human sounds     | ![](https://i.imgur.com/iqorlX3.png) |
|       Vehicles       | ![](https://i.imgur.com/BVOs7x9.png) |
| Household appliances | ![](https://i.imgur.com/0ERrEBV.png) |
|                      |                                      |

### Deep Denoising Autoencoder, DDAE
![](https://i.imgur.com/v45r9Mc.png)



| SNR |             Clean Voice              |                Mixed                 |               Denoised               |
|:---:|:------------------------------------:|:------------------------------------:|:------------------------------------:|
| -10 | ![](https://i.imgur.com/kfS5rWq.png) | ![](https://i.imgur.com/OkkhpRU.png) | ![](https://i.imgur.com/RgtUtfz.png) |
| -5  | ![](https://i.imgur.com/p5aiv8P.png) | ![](https://i.imgur.com/YLUbwpy.png) | ![](https://i.imgur.com/EOkwwrq.png) |
|  0  | ![](https://i.imgur.com/OJ7aGnv.png) | ![](https://i.imgur.com/d0dNDgu.png) | ![](https://i.imgur.com/BsdyVOX.png) |
|  5  | ![](https://i.imgur.com/LA527kY.png) | ![](https://i.imgur.com/hnOHWkz.png) | ![](https://i.imgur.com/hgVe0ey.png) |
| 10  | ![](https://i.imgur.com/P0eWIzy.png) | ![](https://i.imgur.com/EQDW0BS.png) | ![](https://i.imgur.com/GRmGLsB.png) |
| 15  | ![](https://i.imgur.com/cX0rkID.png) | ![](https://i.imgur.com/zsVBgnu.png) | ![](https://i.imgur.com/X5Why2f.png) |