from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('decision_tree_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = [float(data['Max']), float(data['Min']), float(data['Average']), float(data['Ambient']), float(data['Humidity'])]
    prediction = model.predict([features])
    return jsonify({'PMI': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
