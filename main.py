import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import io
import tensorflow as tf
import numpy as np
from tensorflow import keras
from flask import Flask, request, jsonify
from PIL import Image

model = keras.models.load_model('nn')

def transform_image(pillow_image):
    data = np.asarray(pillow_image)
    data = data/255.0
    data = data[np.newaxis, ..., np.newaxis]
    data = tf.image.resize(data, [28, 28])
    return data

def predict(data):
    predictions = model(data)
    predictions = tf.nn.softmax(predictions)
    pred0 = predictions[0]
    label0 = np.argmax(pred0)
    return label0

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        try:
            pillow_image = Image.open(io.BytesIO(file.read())).convert('L')
            data = transform_image(pillow_image)
            label = predict(data)
            return jsonify({'label': label})
        except Exception as e:
            return jsonify({'error': str(e)})
    return "OK"

if __name__ == '__main__':
    app.run(debug=True)
