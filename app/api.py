import joblib
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, request, jsonify, render_template
import pandas as pd
from scripts.model_training import train_model

base_dir = os.path.dirname(os.path.abspath(__file__))
template_folder_path = os.path.join(base_dir, '../templates')
app = Flask(__name__, template_folder=template_folder_path)

# Load dataset and model
cleaned_data_path = os.path.join(base_dir, '../data/cleaned_dataset.csv')
static_dir = os.path.join(base_dir, 'static')
model_path = os.path.join(base_dir, 'customer_support_model.pkl')

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    data = pd.read_csv(cleaned_data_path)
    model = train_model(data)
    joblib.dump(model, model_path)  # Save the model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json

    # Validate Ticket Priority Numeric range
    ticket_priority_numeric = input_data.get('Ticket Priority Numeric')
    if not (1 <= ticket_priority_numeric <= 3):
        return jsonify({"error": "Ticket Priority Numeric must be between 1 and 3."}), 400

    # Prepare features and make a prediction
    features = [[
        input_data['Customer Age'],
        ticket_priority_numeric,
        input_data['Sentiment Score']
    ]]
    prediction = model.predict(features)

    return jsonify({'prediction': bool(prediction[0])})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=True)
