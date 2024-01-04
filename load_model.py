import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.lite.python import schema_py_generated as schema_fb

def crop_image(img):
  ch = img.shape[1] // 2
  cw = img.shape[2] // 2
  k = min(ch, cw)
  return img[:,ch-k:ch+k,cw-k:cw+k,:]

def convert_keras_model(model_name):
    model_names = {
        "resnet50": tf.keras.applications.ResNet50,
        "resnet101": tf.keras.applications.ResNet101,
        "resnet152": tf.keras.applications.ResNet152
    }

    model_fn = model_names.get(model_name)
    if model_fn is None:
        raise ValueError("Unsupported model type")

    model = model_fn(include_top=True, weights='imagenet')
    preprocess_fn = tf.keras.applications.resnet.preprocess_input

    def representative_data_gen():
        for path in os.listdir('samples'):
            img = np.array(Image.open('samples/' + path))
            img = np.expand_dims(img, axis=0)
            if img.shape[-1] != 3:
                continue
            img = tf.image.resize(img, (224, 224))
            yield [preprocess_fn(img)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    tflite_model = converter.convert()

    return tflite_model
