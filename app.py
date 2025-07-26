from flask import Flask, redirect, url_for, request, render_template, session
import sys
import os
import re
import numpy as np
from werkzeug.utils import secure_filename

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img
#from keras.preprocessing.image import load_img
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load model
model_path = 'train_model.h5'
model = load_model(model_path)

app = Flask(__name__)

# Define the directory where uploaded files will be stored
app.config['UPLOAD_FOLDER'] = os.path.join('upload')
app.config['UPLOAD'] = os.path.join('')
# Allow file uploads of up to 16 megabytes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure that file extensions are limited to images only
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

 
@app.route("/")
def main():
    return render_template('main.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')

@app.route("/logout")
def logout():
    return render_template('main.html')


@app.route('/performance')
def performance():
    return render_template('performance.html')


@app.route("/index")
def index():
    return render_template('index.html')

#creates function

def predictions(img_path, model):
    img = load_img(img_path, target_size=(150, 150, 3))

    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    print(x)
    y = model.predict(x)
    print("probabilities:", y)
    pred = np.argmax(model.predict(x)[0], axis=-1)
    print(pred)

    if pred == 0:
        preds = 'YOUNG AGE'
    elif pred == 1:
        preds = 'MIDDLE AGE'
    else:
        preds = 'OLD AGE'

    return preds, y
    print(preds)
    


@app.route("/predicted", methods=['POST'])
def predicted():
    # Get the uploaded image file from the form
    uploaded_file = request.files['imagefile']
    # print(uploaded_file)
    
    # Save the file to a temporary directory
    filename = secure_filename(uploaded_file.filename)
    # print(filename)
    
    #img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # print(img_path)
    img_path = "static/" + filename
    uploaded_file.save(img_path)
    
    img_upload = mpimg.imread(img_path)
    # print(img_upload)

# Add a third dimension to the image array to indicate that it is a grayscale image
    img_upload = np.expand_dims(img_upload, axis=2)
    rgb_image = np.repeat(img_upload, 3, axis=2)
    
    # Save the image to the 'static' directory
    #img = os.path.join('static/savingimg', filename)
    #mpimg.imsave(img, rgb_image)
    # Save the grayscale image to the 'static' directory
    prediction, y = predictions(img_path, model)
    probabilities_string = np.array2string(y, precision=10, separator=',', suppress_small=True)
    
    # Remove the outer square brackets and split the string by comma
    probabilities_list = probabilities_string.strip("[]").split(',')
    
    # Convert the strings to float values
    probabilities = [float(prob.strip()) for prob in probabilities_list]
    probabilities_dict = {
    'YOUNG AGE': probabilities[0],
    'MIDDLE AGE': probabilities[1],
    'OLD AGE': probabilities[2]
    }
    highest_value = (max(y[0]))*100
    
    rounded_number = round(highest_value, 2)
    print(rounded_number)
    # Get the prediction for the uploaded image
    #prediction = predictions(img_path, model)
    
    # Pass the image file path and prediction result to the template
    return render_template('result.html', prediction=prediction, rgb_image=rgb_image, probabilities=rounded_number)


if __name__ == '__main__':
    app.run()
