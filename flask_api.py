# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:50:04 2020

@author: manoj.bahuguna
"""

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import pickle

app = Flask(__name__)
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome Bro how are you"

@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy	= request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "Predicted value is :"+ str(prediction)
    
@app.route('/predict_file', methods = ['POST'])
def predict_note_auth_file():
    df_test = pd.read_csv(request.files.get("files"))
    prediction = classifier.predict(df_test)
    return "Predicted value for csv file is :"+ str(list(prediction))


if __name__ =='__main__':
    app.run()