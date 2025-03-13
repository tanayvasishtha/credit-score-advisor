import numpy as np

def preprocess_input(form_data):
    """
    Preprocess user input from form for model prediction
    """
    # Extract features from form data
    current_score = int(form_data.get('current_score'))
    target_score = int(form_data.get('target_score'))
    monthly_income = float(form_data.get('monthly_income', 0))
    credit_utilization = float(form_data.get('credit_utilization', 0))
    
    # Default values for new features if they're not in the form
    payment_history = float(form_data.get('payment_history', 90))  # Default 90% on-time payments
    credit_age_months = float(form_data.get('credit_age_months', 60))  # Default 5 years
    debt_to_income = float(form_data.get('debt_to_income', 25))  # Default 25% DTI
    
    # Create feature arrays for each model with all features
    X_classifier = np.array([[
        current_score, 
        target_score, 
        monthly_income, 
        credit_utilization,
        payment_history,
        credit_age_months,
        debt_to_income
    ]])
    
    X_regressor = np.array([[
        current_score, 
        target_score, 
        monthly_income, 
        credit_utilization,
        payment_history,
        credit_age_months,
        debt_to_income
    ]])
    
    return X_classifier, X_regressor

def get_recommendations(classifier, X):
    """
    Generate personalized recommendations based on classifier predictions
    """
    # Extract feature values for conditional recommendations
    current_score = X[0][0]
    credit_utilization = X[0][3]
    payment_history = X[0][4]
    credit_age_months = X[0][5]
    debt_to_income = X[0][6]
    
    # Use classifier to predict recommendation category if available
    if classifier is not None:
        recommendation_category = classifier.predict(X)[0]
    else:
        # Fallback if model not loaded
        recommendation_category = 1
    
    recommendations = []
    
    # Basic recommendations for all users
    recommendations.append("Pay all bills on time for the next 3-6 months")
    
    # Credit utilization recommendations
    if credit_utilization > 30:
        recommendations.append(f"Reduce credit card balances from {credit_utilization:.1f}% to below 30% of your limit")
    
    # Payment history recommendations
    if payment_history < 95:
        recommendations.append("Set up automatic payments to avoid missing any future payments")
    
    # Debt to income recommendations
    if debt_to_income > 35:
        recommendations.append(f"Work on reducing your debt-to-income ratio from {debt_to_income:.1f}% to below 35%")
    
    # Credit age recommendations
    if credit_age_months < 24:
        recommendations.append("Keep your oldest credit accounts open to increase your credit history length")
    
    # Score-based recommendations
    if current_score < 670:
        recommendations.append("Avoid opening new credit accounts in the next 6 months")
        recommendations.append("Check your credit report for errors and dispute any inaccuracies")
    elif current_score < 740:
        recommendations.append("Consider diversifying your credit mix with different types of credit")
    
    # Recommendation category specific advice
    if recommendation_category == 0:
        recommendations.append("Focus on reducing overall debt before applying for new credit")
    elif recommendation_category == 1:
        recommendations.append("Maintain current good habits and monitor your credit report regularly")
    elif recommendation_category == 2:
        recommendations.append("You're on a good track - consider requesting credit limit increases on existing accounts")
    
    return recommendations

def predict_timeline(regressor, X):
    """
    Predict the timeline to reach target score
    """
    # Use regressor to predict timeline if available
    if regressor is not None:
        timeline_months = regressor.predict(X)[0]
    else:
        # Fallback calculation if model not loaded
        current_score = X[0][0]
        target_score = X[0][1]
        credit_utilization = X[0][3]
        payment_history = X[0][4]
        
        score_diff = target_score - current_score
        
        # Base estimate: 1 month per 10 points improvement
        timeline_base = score_diff / 10
        
        # Adjust based on credit utilization (high utilization = slower improvement)
        utilization_factor = 1 + (credit_utilization / 100)
        
        # Adjust based on payment history (better history = faster improvement)
        payment_factor = 2 - (payment_history / 100)
        
        # Calculate final estimate
        timeline_months = timeline_base * utilization_factor * payment_factor
    
    return max(1, timeline_months)  # Minimum 1 month
