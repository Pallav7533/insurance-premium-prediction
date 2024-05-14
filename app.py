from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = None
preprocessor = None

def load_model():
    global model
    global preprocessor

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

def preprocess_input(data):
    preprocessed_data = preprocessor.transform(pd.DataFrame(data))
    return preprocessed_data

def predict(data):
    preprocessed_data = preprocess_input(data)
    prediction = model.predict(preprocessed_data)
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    age = int(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    region = request.form['region']

    data = {'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children], 'smoker': [smoker], 'region': [region]}
    prediction = predict(data)

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    load_model() 
    app.run(debug=True)
