# Credit Score Improvement Advisor

A machine learning-powered web application that provides personalized credit score improvement recommendations and timeline predictions.

## 📊 Live Demo
[Credit Score Advisor Live Demo](https://credit-score-advisor.onrender.com/)

## 📝 Project Overview

This Flask-based application uses machine learning to help users improve their credit scores by:
- Analyzing current credit score and financial data
- Generating personalized improvement recommendations
- Predicting the timeline to reach target credit scores
- Evaluating loan eligibility across different credit products

## Technologies Used

<p align="left">
  <a href="https://github.com/tandpfun/skill-icons">
    <img src="https://skillicons.dev/icons?i=python,flask,js,html,css,git" />
  </a>
</p>


## 🧠 Machine Learning Models

### Model 1: Random Forest Classifier
- **Purpose**: Generates personalized credit improvement recommendations
- **Features Used**: Current score, target score, monthly income, credit utilization, payment history, credit age, and debt-to-income ratio
- **Initial Accuracy**: 76.80%
- **Final Accuracy**: 87.45% (after optimization)
- **Improvement Process**:
  - Increased training data from 5,000 to 10,000 samples
  - Added three additional features (payment history, credit age, debt-to-income)
  - Implemented hyperparameter tuning with GridSearchCV
  - Applied class balancing to handle slight imbalances in data
  - Best parameters: `{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}`

### Model 2: Gradient Boosting Regressor
- **Purpose**: Predict the timeline (in months) to reach tthe arget credit score
- **Features Used**: Same as classifier
- **Initial R² Score**: 0.9334
- **Temporary Drop**: 0.7340 (after classifier optimization affected regressor)
- **Final R² Score**: 0.9232 (after separate regressor optimization)
- **Improvement Process**:
  - Implemented RandomizedSearchCV for efficient hyperparameter tuning
  - Optimized key parameters: learning rate, n_estimators, max_depth
  - Best parameters: `{'subsample': 0.9, 'n_estimators': 500, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 6, 'learning_rate': 0.1}`

## 🚀 Features & Functionality

1. **Input Form**:
   - Current credit score
   - Target credit score
   - Monthly income
   - Credit utilization percentage

2. **Personalized Output**:
   - Credit score gap analysis
   - Visual representation via charts
   - Personalized improvement recommendations
   - Timeline prediction
   - Loan eligibility assessment

3. **Technology Stack**:
   - **Frontend**: HTML, CSS, JavaScript, Chart.js
   - **Backend**: Flask (Python)
   - **ML**: scikit-learn, pandas, numpy
   - **Deployment**: Render

## 🛠️ Technical Architecture

```
credit-score-advisor/
├── app.py                  # Main Flask application
├── models/
│   ├── __init__.py         # Makes models a package
│   ├── classifier.pkl      # Trained Random Forest model
│   └── regressor.pkl       # Trained Gradient Boosting model
├── static/
│   ├── css/
│   │   └── style.css       # Main CSS file with financial-themed styling
│   ├── js/
│   │   └── main.js         # JavaScript for form handling and charts
│   └── images/             # Folder for UI images
├── templates/
│   └── index.html          # Single page application template
├── utils/
│   ├── __init__.py         # Makes utils a package
│   ├── data_generator.py   # For creating synthetic credit score data
│   └── model.py            # ML model functions
├── train_models.py         # Script to train and save models
└── requirements.txt        # Project dependencies
```

## 🔄 Development Process & Challenges

### Data Generation
- Created synthetic credit score data with realistic distributions
- Enhanced with additional features like payment history and credit age
- Established logical relationships between features for more realistic data

### Challenges Faced & Solutions
1. **Model Accuracy Improvement**:
   - Initial classifier accuracy was 76.8% 
   - Improved to 87.45% through feature engineering and hyperparameter tuning

2. **Deployment Challenges**:
   - Package compatibility issues with scikit-learn versions
   - Fixed by pinning specific versions in requirements.txt
   - Added Werkzeug compatibility fix for Flask 2.0.1

3. **Model Loading Issues**:
   - Flask scope issues with classifier/regressor variables
   - Fixed by making variables global and adding error handling
   - Implemented fallback recommendations for model loading failures

4. **Styling & UI**:
   - Implemented financial-themed green color scheme
   - Added responsive design for various device sizes
   - Custom-styled eligibility indicators and recommendations

## 📋 Installation & Setup

### Local Development
```bash
# Clone the repository
git clone https://github.com/tanayvasishtha/credit-score-advisor.git
cd credit-score-advisor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train models (optional, pre-trained models included)
python train_models.py

# Run the application
python app.py
```

### Deployment
- Deployed on Render.com using continuous deployment from GitHub
- Used Python 3.11 runtime
- Applied specific package version constraints to ensure compatibility

## 💡 How It Works

1. **Data Generation**: Synthetic credit data models real-world distributions
2. **Model Training**: 
   - Random Forest learns patterns for recommendation categories
   - Gradient Boosting learns to predict improvement timelines
3. **User Input**: Web form collects relevant credit information
4. **Prediction Pipeline**:
   - Data preprocessing
   - Feature extraction
   - Model prediction
   - Results formatting
5. **Visual Presentation**: Results presented in an intuitive, visual format

## 📈 Future Improvements

- Add user authentication for saved profiles
- Implement real credit bureau API integration
- Add historical tracking of credit improvement
- Create more detailed improvement plans with step-by-step guides
- Develop mobile application version

## 👥 Team

-  [ TANAY, PANKAJ, KESHAB ]

---

*"Personalize Your Path to Credit Excellence"*

---
