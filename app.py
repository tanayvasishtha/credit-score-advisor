from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
from utils.model import preprocess_input, get_recommendations, predict_timeline

app = Flask(__name__)

# Load ML models
model_dir = os.path.join(os.path.dirname(__file__), 'models')
try:
    classifier = pickle.load(open(os.path.join(model_dir, 'classifier.pkl'), 'rb'))
    regressor = pickle.load(open(os.path.join(model_dir, 'regressor.pkl'), 'rb'))
    models_loaded = True
except:
    models_loaded = False

@app.route('/')
def home():
    return render_template('index.html', models_loaded=models_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()
        
        # Extract data from form
        current_score = int(data.get('current_score'))
        target_score = int(data.get('target_score'))
        monthly_income = float(data.get('monthly_income', 0))
        credit_utilization = float(data.get('credit_utilization', 0))
        
        # Process data for prediction
        X_classifier, X_regressor = preprocess_input(data)
        
        # Make predictions
        recommendations = get_recommendations(classifier, X_classifier)
        timeline_months = predict_timeline(regressor, X_regressor)
        
        # Determine loan eligibility
        loan_eligibility = {
            "personal_loan": current_score >= 670,
            "car_loan": current_score >= 660,
            "home_loan": current_score >= 740
        }
        
        result = {
            'current_score': current_score,
            'target_score': target_score,
            'score_gap': target_score - current_score,
            'recommendations': recommendations,
            'timeline_months': int(timeline_months),
            'loan_eligibility': loan_eligibility
        }
        
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
