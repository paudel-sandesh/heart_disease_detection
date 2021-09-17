'''# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from HeartDiseases import HeartDisease
import numpy as np
import pickle
import pandas as pd

# 2. Create the app object
app = FastAPI()
pickle_in = open("model_randomforest","rb")
classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Heart Disease Detection'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Heart Disease Detection': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_HeartDisease(data:HeartDisease):
    data = data.dict()
    age=data['age']
    sex=data['sex']
    cp=data['cp']
    trestbps=data['trestbps']	
    chol=data['chol']
    fbs=data['fbs']
    restecg=data['restecg']
    thalach=data['thalach']
    exang=data['exang']
    oldpeak=data['oldpeak']
    slope=data['slope']
    ca=data['ca']
    thal=data['thal']
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak	,slope,ca,thal]])
    if(prediction[0]>0.5):
        prediction="Heart Disease"
    else:
        prediction="No Heart Disease"
    return {
        'prediction': prediction
    }

from fastapi import Form

@app.post("/login/")
async def login(username: str = Form(...), password: str = Form(...)):
    return {"username": username}

@app.post('/predict')
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload'''

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_randomforest', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    #output = round(prediction[0], 2)
    def fxprediction(prediction): 
        if prediction == 1:                
            return(" a heart disease")
        else:
            return("no heart disease")

    return render_template('index.html', prediction_text='patient has {}'.format(fxprediction(prediction)))


if __name__ == "__main__":
    app.run(debug=True)