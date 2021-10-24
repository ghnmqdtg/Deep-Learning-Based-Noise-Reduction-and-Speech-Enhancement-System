# import tensorboard
import tensorflow as tf
import config_params

noise_cls = ["Household_Appliance", "Vechicles"]

for cls in noise_cls:
    model_path = f'./Train/{cls}/weights/DDAE_{config_params.MODEL}_{cls}.h5'
    dest_path = f'./model_files/DDAE_{config_params.MODEL}_{cls}.tflite'

    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Save the converted model to dest_path
    open(dest_path, 'wb').write(converter.convert())
