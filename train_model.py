import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(file_path='gym recommendation.csv'):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {len(df)} rows.")
        return df
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the dataset for recommendation."""
    # Drop ID column as it's not needed for recommendations
    df = df.drop(columns=['ID'])
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # Convert categorical variables to numerical
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
    
    # Define features for similarity
    features = ['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes', 'BMI', 
                'Level', 'Fitness Goal', 'Fitness Type']
    
    # Extract feature matrix
    X = df_processed[features].fillna(0)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df_processed, scaler, features

def save_model(X_scaled, df_processed, scaler, features, output_path='model.pkl'):
    """Save the preprocessed data and scaler for use in the Flask app."""
    model_data = {
        'X_scaled': X_scaled,
        'df_processed': df_processed,
        'scaler': scaler,
        'features': features
    }
    try:
        joblib.dump(model_data, output_path)
        print(f"Model saved successfully to '{output_path}'.")
    except Exception as e:
        print(f"Error saving model: {e}")

def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Preprocess data
    X_scaled, df_processed, scaler, features = preprocess_data(df)
    
    # Save model
    save_model(X_scaled, df_processed, scaler, features)

if __name__ == '__main__':
    main()