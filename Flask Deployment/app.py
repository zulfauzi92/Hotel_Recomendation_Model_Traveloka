# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 22:30:42 2022

@author: admin
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd
from ast import literal_eval

import tensorflow as tf

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
import requests

# df_user = pd.read_csv('https://raw.githubusercontent.com/zulfauzi92/Hotel_Recomendation_Model_Traveloka/main/Eksplorasi%20Data/Main%20Dataset/csv_final//Final_Dataset_User_ML.csv', index_col=[0])
# df_hotel_ML = pd.read_csv('https://raw.githubusercontent.com/zulfauzi92/Hotel_Recomendation_Model_Traveloka/main/Eksplorasi%20Data/Main%20Dataset/csv_final//Final_Dataset_Hotel_ML.csv', index_col=[0])
# df_hotel_nonML = pd.read_csv('https://raw.githubusercontent.com/zulfauzi92/Hotel_Recomendation_Model_Traveloka/main/Eksplorasi%20Data/Main%20Dataset/csv_final//Final_Dataset_Hotel_nonML.csv', index_col=[0])
# df_review = pd.read_csv('https://raw.githubusercontent.com/zulfauzi92/Hotel_Recomendation_Model_Traveloka/main/Eksplorasi%20Data/Main%20Dataset/csv_final//Final_Dataset_Review.csv', index_col=[0])
df_hotel_ML_transformed = pd.read_csv('https://raw.githubusercontent.com/zulfauzi92/Hotel_Recomendation_Model_Traveloka/main/Recomendation%20Model/ML_user_data_transformed.csv', index_col=[0])

df_hotel_ML_transformed['Item_id'] = df_hotel_ML_transformed['Item_id'].apply(lambda x: str(np.array(literal_eval(x))[0]))

# Define a flask app
app = Flask(__name__,template_folder='templates')

# Model saved with Keras model.save()
MODEL_PATH = 'C:/Users/admin/Documents\GitHub\Hotel_Recomendation_Model_Traveloka\Recomendation Model\keras_h5\collaborative_model.h5'

model = load_model(MODEL_PATH)
model.make_predict_function()

print('Model loaded. Check http://127.0.0.1:3000/')

# Contoh Input dari BE
user_input = 'U09081'


def convertListfromInteger(identifier, list_int):
    converted = []
    for n in list_int:
        converted.append(identifier+'{0:06}'.format(n))
    return converted

def convertIntegerfromList(list_string):
    converted = []
    for n in list_string:
        converted.append(int(n[1:]))
    return converted

def model_predict(array, model):
    preds = model.predict(array)
    return preds

@app.route('/', methods=['GET','POST'])
def main():
    # menerima input json
    # request_data = request.get_json() # line di comment untuk testing
    
    # mengubah string hotel_id ke integer
    arr_hotel = convertIntegerfromList(df_hotel_ML_transformed['Item_id'].values)
    # membuat array user sejumlah hotel
    arr_user = np.full(shape=len(arr_hotel), fill_value=int(user_input[1:]), dtype=np.int64)

    preds = model_predict([tf.constant(arr_user),tf.constant(arr_hotel)], model)
    predictions  = np.array([a[0] for a in preds])
    recommended_item_ids = (-predictions).argsort()[:len(arr_hotel)]
    
    arr_output = []
    for item in recommended_item_ids:
        arr_output.append(arr_hotel[item])

    output = convertListfromInteger('H', arr_output)
    # print(len(output))
    dict_output = {
        'recommended_id':output
    }
    
    # return jsonify(dict_output)
    return render_template('index.html', prediction=output)
    
      
if __name__ == '__main__':
    app.run(port=3000, debug=False)