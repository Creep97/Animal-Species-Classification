from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

new_model = tf.keras.models.load_model('saved_model/my_model')

CATEGORIES = ['African Elephant','Amur Leopard','Arctic Fox','Black Rhino','Black Spider Monkey','Bluefin Tuna','Brown Bear','Chimpanzee','European Rabbit','Orangutan']

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256,256,3))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    images=np.vstack([x])
    preds = new_model.predict(x)
    result = CATEGORIES[np.argmax(preds)]
    print(result)
    return preds   


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, new_model)
        result = CATEGORIES[np.argmax(preds)]              # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)