# -*- coding: utf-8 -*-
"""
@author: prana
"""
import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import pathlib

#model=pickle.load(open("model.pkl", 'rb'))
app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/usecloud")
def usecloud():
    return render_template("use_cloud.html")

@app.route("/uselocal")
def uselocal():
    return render_template("use_local.html")

data_dir = "C:/Users/prana/CustomDataTrainer/FolderTrainer/Training"
val_dir= "C:/Users/prana/CustomDataTrainer/FolderTrainer/Validation"

@app.route("/predict",methods=["POST","GET"])
def predict():

    img_height,img_width=180,180
    batch_size=32
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.1,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    print("Classes loaded")
    class_names = train_ds.class_names
    no_of_classes=len(class_names)
    print("Model build Start")
    vgg_model = Sequential()

    pretrained_model= tf.keras.applications.VGG16(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',classes=no_of_classes,
                   weights='imagenet')
    for layer in pretrained_model.layers[:-23]:
        layer.trainable=False

    vgg_model.add(pretrained_model)
    vgg_model.add(Flatten()) 
    vgg_model.add(Dense(512, activation='relu'))
    vgg_model.add(Dense(10, activation='softmax'))
    vgg_model.summary()

    vgg_model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    print("Epoch starts")
    epochs=1
    history = vgg_model.fit(train_ds,validation_data=val_ds,epochs=epochs)
    
    
# =============================================================================
#     inps=[float(request.form.get('hair_length')),float(request.form.get('lname')),float(request.form.get('fname')),float(request.form.get('nose_wide')),float(request.form.get('nose_long')),float(request.form.get('lips_th')),float(request.form.get('dist_lip'))]
#     prediction=model.predict([inps])
            
#     param0=inps[0] 
#     param1=inps[1]   
#     param2=inps[2]   
#     param3=inps[3] 
#     param4=inps[4]   
#     param5=inps[5]  
#     param6=inps[6]  

#     return render_template('index.html',prediction_text="Result predicted by model: ",
#                 param1="Forehead Width: "+str(param1),
#                 param0="Hair Long: "+str(param0),
#                 param2="Forehead Height: "+str(param2),
#                 param3="Nose wide: "+str(param3),
#                 param4="Nose long: "+str(param4),
#                 param5="Lips thin: "+str(param5),
#                 param6="Long distance between lips and nose: "+str(param6)
#                 )
# =============================================================================

    return render_template('index.html',noofclasses="Number of classes found are: "+str(no_of_classes),
                           nameofclasses="Categories: "+str(class_names))

if __name__ =="__main__":
    app.run(debug=True)

