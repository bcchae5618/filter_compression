import tensorflow as tf
import tensorflow_datasets as tfds
from load_model import crop_image

def evaluate(model_path):
  interpreter = tf.lite.Interpreter(model_path)
  interpreter.allocate_tensors()

  dataset = tfds.load('imagenet2012', split='validation').take(100).batch(1)
  acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

  preprocess_fn = tf.keras.applications.resnet.preprocess_input

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  input_scale, input_zero_point = input_details['quantization']
  output_scale, output_zero_point = output_details['quantization']

  # Run a testidation loop at the end of each epoch.
  for batch in dataset:
    x_batch = tf.image.resize(batch["image"], (224, 224))
    x_batch = preprocess_fn(tf.cast(x_batch, dtype=tf.float32))
    y_batch = batch["label"]

    # Quantize the input tensor
    if (input_scale, input_zero_point) != (0.0, 0):
      x_batch = x_batch / input_scale + input_zero_point
      x_batch = tf.cast(x_batch, dtype=input_details['dtype'])

    interpreter.set_tensor(input_details['index'], x_batch)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details['index'])

    if (output_scale, output_zero_point) != (0.0, 0):
      # Dequantize the output tensor
      logits = (logits - output_zero_point) * output_scale
      logits = tf.cast(logits, dtype=output_details['dtype'])

    acc_metric.update_state(y_batch, logits)

  acc = acc_metric.result()
  acc_metric.reset_states()
  print("Accuracy: %.4f" % (float(acc),))
