import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

app = Flask(__name__)

# Load and preprocess the dataset
def load_data():
    try:
        df = pd.read_csv('gym recommendation(1).csv')
        # Drop ID column as it's not needed for recommendations
        df = df.drop(columns=['ID'])
        return df
    except FileNotFoundError:
        print("Error: 'gym recommendation.csv' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Preprocess data for recommendation
def preprocess_data(df):
    # Convert categorical variables to numerical
    df_processed = df.copy()
    df_processed['Sex'] = df_processed['Sex'].map({'Male': 1, 'Female': 0})
    df_processed['Hypertension'] = df_processed['Hypertension'].map({'Yes': 1, 'No': 0})
    df_processed['Diabetes'] = df_processed['Diabetes'].map({'Yes': 1, 'No': 0})
    df_processed['Level'] = df_processed['Level'].map({
        'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obuse': 3
    })
    df_processed['Fitness Goal'] = df_processed['Fitness Goal'].map({
        'Weight Gain': 0, 'Weight Loss': 1
    })
    df_processed['Fitness Type'] = df_processed['Fitness Type'].map({
        'Muscular Fitness': 0, 'Cardio Fitness': 1
    })
    
    # Features for similarity
    features = ['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes', 'BMI', 
                'Level', 'Fitness Goal', 'Fitness Type']
    X = df_processed[features].fillna(0)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df_processed, scaler, features

# Get recommendation based on user input
def get_recommendation(user_input, X_scaled, df_processed, scaler, features):
    # Create user input DataFrame
    user_df = pd.DataFrame([user_input], columns=features)
    
    # Convert categorical variables
    user_df['Sex'] = user_df['Sex'].map({'Male': 1, 'Female': 0})
    user_df['Hypertension'] = user_df['Hypertension'].map({'Yes': 1, 'No': 0})
    user_df['Diabetes'] = user_df['Diabetes'].map({'Yes': 1, 'No': 0})
    user_df['Level'] = user_df['Level'].map({
        'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obuse': 3
    })
    user_df['Fitness Goal'] = user_df['Fitness Goal'].map({
        'Weight Gain': 0, 'Weight Loss': 1
    })
    user_df['Fitness Type'] = user_df['Fitness Type'].map({
        'Muscular Fitness': 0, 'Cardio Fitness': 1
    })
    
    # Scale user input
    user_scaled = scaler.transform(user_df.fillna(0))
    
    # Calculate similarity
    similarities = cosine_similarity(user_scaled, X_scaled)
    best_match_idx = np.argmax(similarities)
    
    # Return recommendation
    return df_processed.iloc[best_match_idx][['Exercises', 'Equipment', 'Diet', 'Recommendation']]

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendation = None
    error = None
    
    # Load data
    df = load_data()
    if df is None:
        return render_template('index.html', error="Dataset not found.")
    
    # Preprocess data
    X_scaled, df_processed, scaler, features = preprocess_data(df)
    
    if request.method == 'POST':
        try:
            # Get user input
            user_input = {
                'Sex': request.form['sex'],
                'Age': float(request.form['age']),
                'Height': float(request.form['height']),
                'Weight': float(request.form['weight']),
                'Hypertension': request.form['hypertension'],
                'Diabetes': request.form['diabetes'],
                'BMI': float(request.form['weight']) / (float(request.form['height']) ** 2),
                'Level': request.form['level'],
                'Fitness Goal': request.form['fitness_goal'],
                'Fitness Type': request.form['fitness_type']
            }
            
            # Get recommendation
            recommendation = get_recommendation(user_input, X_scaled, df_processed, scaler, features)
            recommendation = recommendation.to_dict()
        except Exception as e:
            error = f"Error processing input: {str(e)}"
    
    return render_template('index.html', recommendation=recommendation, error=error)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
