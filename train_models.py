import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from utils.data_generator import generate_credit_score_data
from sklearn.utils import class_weight

def train_and_save_models():
    """
    Train ML models and save them as pickle files
    """
    print("Generating synthetic credit score data...")
    df_class, df_reg = generate_credit_score_data(n_samples=10000)  # Increased from 5000 to 10000
    
    # Classification model (for recommendations)
    X_class = df_class.drop('recommendation_category', axis=1)
    y_class = df_class['recommendation_category']
    
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )
    
    # Check class distribution
    from collections import Counter
    class_counts = Counter(y_train_class)
    print(f"Class distribution: {class_counts}")
    
    # Calculate class weights for balancing
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train_class),
        y=y_train_class
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print("Computed class weights:", class_weight_dict)
    
    print("Training Random Forest Classifier with GridSearchCV for optimal parameters...")
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Create grid search with cross-validation
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight=class_weight_dict),
        param_grid=param_grid,
        cv=3,  # Using 3-fold CV to save time
        scoring='accuracy',
        n_jobs=-1  # Use all available cores
    )
    
    # Fit grid search
    grid_search.fit(X_train_class, y_train_class)
    
    # Get best parameters and model
    best_params = grid_search.best_params_
    classifier = grid_search.best_estimator_
    
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    class_accuracy = classifier.score(X_test_class, y_test_class)
    print(f"Classifier accuracy on test set: {class_accuracy:.4f}")
    
    # Feature importance
    print("\nFeature Importance:")
    for feature, importance in zip(X_class.columns, classifier.feature_importances_):
        print(f"{feature}: {importance:.4f}")
    
    # Regression model (for timeline prediction)
    X_reg = df_reg.drop('timeline_months', axis=1)
    y_reg = df_reg['timeline_months']
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    print("\nOptimizing Gradient Boosting Regressor with RandomizedSearchCV...")
    
    # Define parameter distribution for RandomizedSearchCV
    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=20,  # Number of parameter settings sampled
        cv=3,       # Cross-validation folds
        scoring='r2',
        n_jobs=-1,  # Use all available cores
        random_state=42
    )
    
    # Fit RandomizedSearchCV
    random_search.fit(X_train_reg, y_train_reg)
    
    # Get best parameters and model
    best_params = random_search.best_params_
    regressor = random_search.best_estimator_
    
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation R² score: {random_search.best_score_:.4f}")
    
    # Evaluate on test set
    reg_score = regressor.score(X_test_reg, y_test_reg)
    print(f"Regressor R² score on test set: {reg_score:.4f}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save models
    print("Saving models...")
    with open('models/classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    with open('models/regressor.pkl', 'wb') as f:
        pickle.dump(regressor, f)
    
    print("Models trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_models()
