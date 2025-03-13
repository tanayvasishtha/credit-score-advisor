import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

def generate_credit_score_data(n_samples=10000):
    """
    Generate synthetic credit score data for training models
    """
    # Create features for classification with more informative features
    X_class, y_class = make_classification(
        n_samples=n_samples,
        n_features=7,  # Increased from 4 to 7
        n_informative=5,  # Increased from 3 to 5
        n_redundant=1,
        n_classes=3,
        class_sep=1.2,  # Increased class separation
        random_state=42
    )
    
    # Create features for regression (timeline prediction)
    X_reg, y_reg = make_regression(
        n_samples=n_samples,
        n_features=7,  # Increased from 4 to 7
        n_informative=5,  # Increased from 3 to 5
        noise=5,  # Reduced noise from 10 to 5
        random_state=42
    )
    
    # Scale y_reg to represent months (between 1-36 months)
    y_reg = np.abs(y_reg)
    y_reg = (y_reg - y_reg.min()) / (y_reg.max() - y_reg.min()) * 35 + 1
    
    # Create feature names with additional features
    feature_names = [
        'current_score', 
        'target_score', 
        'monthly_income', 
        'credit_utilization',
        'payment_history',  # New feature
        'credit_age_months',  # New feature
        'debt_to_income'  # New feature
    ]
    
    # Scale features to realistic ranges
    X_class_scaled = X_class.copy()
    X_reg_scaled = X_reg.copy()
    
    # Current score: 300-850
    X_class_scaled[:, 0] = (X_class[:, 0] - X_class[:, 0].min()) / (X_class[:, 0].max() - X_class[:, 0].min()) * 550 + 300
    X_reg_scaled[:, 0] = (X_reg[:, 0] - X_reg[:, 0].min()) / (X_reg[:, 0].max() - X_reg[:, 0].min()) * 550 + 300
    
    # Target score: slightly higher than current score
    X_class_scaled[:, 1] = X_class_scaled[:, 0] + np.abs(X_class[:, 1]) * 100 + 20
    X_reg_scaled[:, 1] = X_reg_scaled[:, 0] + np.abs(X_reg[:, 1]) * 100 + 20
    
    # Cap target score at 850
    X_class_scaled[:, 1] = np.minimum(X_class_scaled[:, 1], 850)
    X_reg_scaled[:, 1] = np.minimum(X_reg_scaled[:, 1], 850)
    
    # Monthly income: $2000-$15000
    X_class_scaled[:, 2] = (X_class[:, 2] - X_class[:, 2].min()) / (X_class[:, 2].max() - X_class[:, 2].min()) * 13000 + 2000
    X_reg_scaled[:, 2] = (X_reg[:, 2] - X_reg[:, 2].min()) / (X_reg[:, 2].max() - X_reg[:, 2].min()) * 13000 + 2000
    
    # Credit utilization: 0-100%
    X_class_scaled[:, 3] = (X_class[:, 3] - X_class[:, 3].min()) / (X_class[:, 3].max() - X_class[:, 3].min()) * 100
    X_reg_scaled[:, 3] = (X_reg[:, 3] - X_reg[:, 3].min()) / (X_reg[:, 3].max() - X_reg[:, 3].min()) * 100
    
    # Payment history: 0-100% (percentage of on-time payments)
    X_class_scaled[:, 4] = (X_class[:, 4] - X_class[:, 4].min()) / (X_class[:, 4].max() - X_class[:, 4].min()) * 100
    X_reg_scaled[:, 4] = (X_reg[:, 4] - X_reg[:, 4].min()) / (X_reg[:, 4].max() - X_reg[:, 4].min()) * 100
    
    # Credit age in months: 1-360 months (1-30 years)
    X_class_scaled[:, 5] = (X_class[:, 5] - X_class[:, 5].min()) / (X_class[:, 5].max() - X_class[:, 5].min()) * 359 + 1
    X_reg_scaled[:, 5] = (X_reg[:, 5] - X_reg[:, 5].min()) / (X_reg[:, 5].max() - X_reg[:, 5].min()) * 359 + 1
    
    # Debt to income ratio: 0-60%
    X_class_scaled[:, 6] = (X_class[:, 6] - X_class[:, 6].min()) / (X_class[:, 6].max() - X_class[:, 6].min()) * 60
    X_reg_scaled[:, 6] = (X_reg[:, 6] - X_reg[:, 6].min()) / (X_reg[:, 6].max() - X_reg[:, 6].min()) * 60
    
    # Create logical relationships between features to make dataset more realistic
    
    # Lower credit scores tend to have higher credit utilization
    for i in range(n_samples):
        if X_class_scaled[i, 0] < 650:  # For low credit scores
            X_class_scaled[i, 3] = min(X_class_scaled[i, 3] + 20, 100)  # Increase utilization
            
        if X_reg_scaled[i, 0] < 650:  # For low credit scores
            X_reg_scaled[i, 3] = min(X_reg_scaled[i, 3] + 20, 100)  # Increase utilization
    
    # Create dataframes
    df_class = pd.DataFrame(X_class_scaled, columns=feature_names)
    df_class['recommendation_category'] = y_class
    
    df_reg = pd.DataFrame(X_reg_scaled, columns=feature_names)
    df_reg['timeline_months'] = y_reg
    
    return df_class, df_reg

if __name__ == "__main__":
    # Generate and save data
    df_class, df_reg = generate_credit_score_data()
    df_class.to_csv('classification_data.csv', index=False)
    df_reg.to_csv('regression_data.csv', index=False)
    print("Synthetic data generated and saved successfully!")
