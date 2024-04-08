from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('insurance_cost_model.sav', 'rb'))
encoders = pickle.load(open('insurance_cost_encoders.sav', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Get form data
    data = {}
    data['age'] = float(request.form.get('age'))  
    data['sex'] = request.form.get('sex')
    data['bmi'] = float(request.form.get('bmi'))  
    data['children'] = int(request.form.get('children')) 
    data['smoker'] = request.form.get('smoker')
    data['region'] = request.form.get('region')



    data['sex'] = encoders['sex'].transform([data['sex']])[0]


    data['smoker'] = encoders['smoker'].transform([data['smoker']])[0]

    df = pd.DataFrame([data])


    for i in encoders['region'].categories_[0]:
        df['region' + '_' + i] = 0.0
    df['region' + '_' + df['region']] = 1.0
    df.drop(columns='region', inplace=True)


    pred = model.predict(df)[0].round(3)
    return render_template('index.html', prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)