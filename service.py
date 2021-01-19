# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 19:18:22 2020

@author: ahmad
"""

from flask import Flask, request, jsonify, render_template, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import base64
import io
from gradcam.code import GradCAM
import cv2
import scipy as sp
import os

app = Flask(__name__)

def deltemp():
    try:
        os.remove('temp/OVL_2/0.png')
        os.remove('temp/IM/0.png')
        os.remove('temp/OVL/0.png')
    except:
        pass
    return print('temp file removed')

@app.route('/index')
def index():
   deltemp()
   return render_template('index.html')

def loadmodel():
    model = load_model('vgg19gap_checkpoint.h5')
    return model

def data():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    return image
    
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image).astype('float32')*1/255
    image = np.expand_dims(image,axis=0)
    return image

def result(image):
    processed_image = preprocess_image(image, target_size=(224, 224))
    prediction = model.predict(processed_image)
    return prediction, processed_image

def gradcam():
    model = load_model()
    image = data()
    prediction = result(image)[0]
    processed_image = result(image)[1]
    test_image_gradcam = prediction
    test_image_expand = processed_image
    i = np.argmax(test_image_gradcam[0])
    cam = GradCAM(model, i)
    heatmap = cam.compute_heatmap(test_image_expand)
    return heatmap  

model = loadmodel()

@app.route('/predict', methods=['POST'])
def predict():
    image = data()
    
    prediction = result(image)[0]
    
    test_image_probs = {prediction[0][0] * 100 : "Pleural Effusion" ,
                        prediction[0][1] * 100 : "Normal Lungs"}
    
    test_image_probs = {v: k for k, v in sorted(test_image_probs.items(), reverse=True)}
    
    testimage_prob_sorted_keys = list(test_image_probs.keys())
    testimage_prob_sorted_values = list(test_image_probs.values())

    prob_key_1 = testimage_prob_sorted_keys[0]
    prob_value_1 = round(testimage_prob_sorted_values[0], 2)

    prob_key_2 = testimage_prob_sorted_keys[1]
    prob_value_2 = round(testimage_prob_sorted_values[1], 2)
    
    response = {
        'prediction' : {    
            'prob_key_1' : prob_key_1,
            'prob_value_1' : prob_value_1,
            'prob_key_2' : prob_key_2,
            'prob_value_2' : prob_value_2
            }
        }
    
    # gradcam 1
    processed_image = result(image)[1]
    i = np.argmax(prediction[0])
    cam = GradCAM(model, i)
    heatmap = cam.compute_heatmap(processed_image)
    
    width, height = 1024, 1024
    
    heatmap = sp.ndimage.zoom(heatmap, (width/224, height/224), order=1)
    heatmap = np.array(heatmap)
    heatmap = cv2.resize(heatmap, (width, height))
    heatmap = np.expand_dims(heatmap,axis=2)
    heatmap= cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)      
    heatmap = Image.fromarray(heatmap.astype('uint8'))
    
    background = np.array(image)
    background = cv2.resize(background, (width, height))
    back_shape = np.shape(background)
    if back_shape == (1024, 1024, 4):
        background = cv2.cvtColor(background, cv2.COLOR_RGBA2RGB)
        background = Image.fromarray(background.astype('uint8'))
        background.save('temp/IM/0.png',"PNG")
        overlayed = Image.blend(background, heatmap, 0.3)
    else :
        background = np.expand_dims(background,axis=2)
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
        background = Image.fromarray(background.astype('uint8'))
        background.save('temp/IM/0.png',"PNG")
        overlayed = Image.blend(background, heatmap, 0.3)
    overlayed.save('temp/OVL/0.png',"PNG")
    
    # gradcam 2
    processed_image = result(image)[1]
    if i == 0:
        j = 1
    else :
        j = 0
    cam_2 = GradCAM(model, j)
    heatmap_2 = cam_2.compute_heatmap(processed_image)
    
    width, height = 1024, 1024
    
    heatmap_2 = sp.ndimage.zoom(heatmap_2, (width/224, height/224), order=1)
    heatmap_2 = np.array(heatmap_2)
    heatmap_2 = cv2.resize(heatmap_2, (width, height))
    heatmap_2 = np.expand_dims(heatmap_2,axis=2)
    heatmap_2 = cv2.cvtColor(heatmap_2, cv2.COLOR_GRAY2BGR)
    heatmap_2 = cv2.applyColorMap(heatmap_2, cv2.COLORMAP_JET)
    heatmap_2 = cv2.cvtColor(heatmap_2, cv2.COLOR_BGR2RGB)      
    heatmap_2 = Image.fromarray(heatmap_2.astype('uint8'))
    
    background_2 = np.array(image)
    background_2 = cv2.resize(background_2, (width, height))
    back_shape_2 = np.shape(background_2)
    if back_shape_2 == (1024, 1024, 4):
        background_2 = cv2.cvtColor(background_2, cv2.COLOR_RGBA2RGB)
        background_2 = Image.fromarray(background_2.astype('uint8'))
        background_2.save('temp/IM/0.png',"PNG")
        overlayed_2 = Image.blend(background_2, heatmap_2, 0.3)
    else :
        background_2 = np.expand_dims(background_2,axis=2)
        background_2 = cv2.cvtColor(background_2, cv2.COLOR_GRAY2RGB)
        background_2 = Image.fromarray(background_2.astype('uint8'))
        background_2.save('temp/IM/0.png',"PNG")
        overlayed_2 = Image.blend(background_2, heatmap_2, 0.3)
    overlayed_2.save('temp/OVL_2/0.png',"PNG")
                
    return jsonify(response)

@app.route('/Primary-Diagnosis')
def ovl():
    return send_file('temp/OVL/0.png')

@app.route('/Other-Diagnosis')
def ovl_2():
    return send_file('temp/OVL_2/0.png')

@app.route('/x-ray')
def xray():
    return send_file('temp/IM/0.png')

app.run(host='0.0.0.0',port=5000)

#http://127.0.0.1:5000/index