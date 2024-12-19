import joblib
import os
import sys
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, request, jsonify, render_template
import pandas as pd
from scripts.model_training import train_model

app = Flask(__name__, template_folder='/Users/dreytee/PycharmProjects/LLM_Ticket_Insights/templates')

# Load dataset and model
model_path = '/Users/dreytee/PycharmProjects/LLM_Ticket_Insights/customer_support_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    data = pd.read_csv('/Users/dreytee/PycharmProjects/LLM_Ticket_Insights/data/cleaned_dataset.csv')
    model = train_model(data)
    joblib.dump(model, model_path)  # Save the model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    features = [[input_data['Customer Age'], input_data['Ticket Priority Numeric'], input_data['Sentiment Score']]]

    prediction = model.predict(features)
    return jsonify({'prediction': bool(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
