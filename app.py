import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

app = Flask(__name__)

# --- GLOBAL VARIABLES ---
MODEL = None
DATA_PATH = 'Manufacturing_ESG_Financial_Data.csv'

# --- MODEL TRAINING & LOADING ---
def load_or_train_model():
    """Trains the model if it doesn't exist, otherwise loads it."""
    global MODEL
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return None

    print("Loading data and training model...")
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Features and Target
        target = 'E_Score'
        # Features available for prediction + target components
        features = ['Industry_Type', 'Firm_Size', 'Revenue', 'S_Score', 'G_Score', 'ESG_Score']
        
        X = df[features]
        y = df[target]

        # Preprocessing
        categorical_cols = ['Industry_Type']
        numerical_cols = [c for c in features if c not in categorical_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])

        # Pipeline
        MODEL = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        MODEL.fit(X, y)
        print("Model trained successfully.")
        return MODEL
    except Exception as e:
        print(f"Error during training: {e}")
        return None

# Initial Training on Startup
MODEL = load_or_train_model()

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({'error': 'Model not trained. Check if CSV file exists.'}), 500

    try:
        # Get data from AJAX request
        data = request.json
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame({
            'Industry_Type': [data['industry']],
            'Firm_Size': [float(data['firm_size'])],
            'Revenue': [float(data['revenue'])],
            'S_Score': [float(data['s_score'])],
            'G_Score': [float(data['g_score'])],
            'ESG_Score': [float(data['esg_score'])]
        })

        # Predict
        prediction = MODEL.predict(input_df)[0]
        
        # Interpret result for UI badges
        status = "High Risk"
        color = "danger"
        if prediction > 50:
            status = "Average"
            color = "warning"
        if prediction > 75:
            status = "Excellent"
            color = "success"

        return jsonify({
            'prediction': round(prediction, 2),
            'status': status,
            'color': color
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)