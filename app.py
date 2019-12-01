#Define Dependencies
from flask import Flask, render_template, request
import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import json
import cv2
import os
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import sys
import matplotlib.pyplot as plt
app = Flask(__name__) #initializing flask app

#loading our saved model
with open('model.json','r') as f:
    model_loaded = model_from_json(f.read())

model_loaded.load_weights('model.h5')
model_loaded.summary()

model_loaded.compile('adam', 'categorical_crossentropy', metrics = ['acc'])



#name = 'pitambar'
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/', methods = ['POST'])
#Defining the function that is used to save the uploaded image and also read the image and also predict the uploaded image 
def predict():
	if request.method == 'POST':
	    f = request.files['file'] #get the form data.
	    basepath = os.path.dirname(__file__)
	    file_path = os.path.join(secure_filename(f.filename))
	    f.save(file_path)

	img = cv2.imread(file_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	plt.imshow(img)

	img1 = cv2.resize(img, (28, 28))
	img1 = img1.reshape(-1, 28, 28, 1)/255
	prediction = model_loaded.predict(img1)
	pred = np.argmax(prediction)
	return render_template('predict.html', x = pred)

if __name__ == '__main__':
    app.run()   