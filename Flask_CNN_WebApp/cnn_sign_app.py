from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras
import pandas as pd
import cv2


app = Flask(__name__)

model = keras.models.load_model(r'Sign_CNN.h5')
model.load_weights(r'Sign_CNN.h5')


labels = pd.read_csv("labels.csv").values


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        img = request.files['images']
        img.save("img.jpg")
        
        image = cv2.imread("img.jpg")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = cv2.resize(image, (90,90))
        
        image = np.reshape(image, (1,90,90,3))
        
        pred = model.predict(image)
        
        pred = np.argmax(pred)
 
        pred = labels[pred]              
        return render_template('predict.html', prediction = pred)
    return None

@app.route('/')
def start():
    return render_template('startpage.html')

@app.route('/index')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
