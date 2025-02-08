import json
import pickle

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  # If using a scaler

app = Flask(__name__)

# Load the trained logistic regression model
logimodel = pickle.load(open('logisticmodel.pkl', 'rb'))

# If using a scaler, load it similarly
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

app = Flask(__name__)

# Load the trained logistic regression model
logimodel = pickle.load(open('logisticmodel.pkl', 'rb'))

# If using a scaler, load it similarly
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()  # Fetch form data
    print(data)

    # Convert input data to numpy array and reshape
    data_values = np.array(list(map(float, data.values()))).reshape(1, -1)

    # Transform the data using the scaler
    new_data = scalar.transform(data_values)

    # Make a prediction
    output = logimodel.predict(new_data)[0]

    return render_template('home.html', prediction_text=f'Predicted Outcome: {output}')

if __name__ == '__main__':
    app.run(debug=True)
