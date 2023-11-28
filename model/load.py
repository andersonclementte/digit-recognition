from tensorflow import keras
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# model = keras.models.load_model('nn')
model = keras.models.load_model('nn.h5')

with open('./model/three.png', 'rb') as file:
    image_bytes = file.read()
    pillow_image = Image.open(io.BytesIO(image_bytes)).convert('L')

data = np.asarray(pillow_image)
data = data/255.0
data = data[np.newaxis, ..., np.newaxis]
data = tf.image.resize(data, [28, 28])

predictions = model(data)
predictions = tf.nn.softmax(predictions)
pred0 = predictions[0]
label0 = np.argmax(pred0)
print(label0)