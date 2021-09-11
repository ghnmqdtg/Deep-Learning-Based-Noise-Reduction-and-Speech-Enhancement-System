# import tensorboard
import tensorflow as tf
from tensorflow.keras.models import load_model

# print(tf.__version__)
# print(help(tf.lite.TFLiteConverter))

model_path = './DDAE_Household/Train/weights/model_DDAE.h5'
dest_path = './DDAE_Household/Train/weights/model.tflite'
model = load_model(model_path)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(dest_path, 'wb') as f:
    f.write(tflite_model)
