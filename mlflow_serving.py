import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

class MLFlowModelServer:
    def __init__(self, model_path=None):
        """
        Initialize MLFlow model server
        """
        self.model = None
        self.scaler = None
        
        if model_path:
            self.load_model(model_path)
    
    def save_model(self, model, scaler, model_name="rossmann_sales_model"):
        """
        Save model using MLFlow
        """
        with mlflow.start_run():
            # Log model
            mlflow.sklearn.log_model(model, model_name)
            
            # Save scaler separately
            scaler_path = f"models/scaler_{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.pkl"
            joblib.dump(scaler, scaler_path)
            
            # Log scaler as artifact
            mlflow.log_artifact(scaler_path)
            
            # Log model metrics (if available)
            mlflow.log_param("model_type", type(model).__name__)
            
            print(f"Model saved successfully with MLFlow")
            
    def load_model(self, model_path):
        """
        Load model from MLFlow
        """
        try:
            self.model = mlflow.sklearn.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def load_scaler(self, scaler_path):
        """
        Load scaler
        """
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded successfully")
        except Exception as e:
            print(f"Error loading scaler: {e}")
    
    def preprocess_input(self, data):
        """
        Preprocess input data for prediction
        """
        # Convert date columns if present
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data['Year'] = data['Date'].dt.year
            data['Month'] = data['Date'].dt.month
            data['Day'] = data['Date'].dt.day
            data['DayOfWeek'] = data['Date'].dt.dayofweek
            data['WeekOfYear'] = data['Date'].dt.isocalendar().week
            
        # Create additional features
        data['IsWeekend'] = data.get('DayOfWeek', 0).apply(lambda x: 1 if x >= 5 else 0)
        
        # Handle missing values
        data = data.fillna(0)
        
        # Select features used in training (adjust based on your model)
        feature_columns = [
            'Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday_encoded',
            'SchoolHoliday', 'StoreType_encoded', 'Assortment_encoded',
            'CompetitionDistance', 'Promo2', 'Year', 'Month', 'Day',
            'WeekOfYear', 'IsWeekend'
        ]
        
        # Keep only available columns
        available_columns = [col for col in feature_columns if col in data.columns]
        processed_data = data[available_columns]
        
        # Scale data if scaler is available
        if self.scaler:
            processed_data = self.scaler.transform(processed_data)
            
        return processed_data
    
    def predict(self, data):
        """
        Make predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Preprocess data
        processed_data = self.preprocess_input(data)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        
        return predictions

# Example usage
if __name__ == "__main__":
    # Initialize server
    server = MLFlowModelServer()
    
    # Example: Load a saved model
    # server.load_model("models:/rossmann_sales_model/1")
    # server.load_scaler("models/scaler.pkl")
    
    # Example: Make prediction
    sample_data = pd.DataFrame({
        'Store': [1],
        'Date': ['2024-01-01'],
        'DayOfWeek': [1],
        'Open': [1],
        'Promo': [0],
        'StateHoliday_encoded': [0],
        'SchoolHoliday': [0],
        'StoreType_encoded': [1],
        'Assortment_encoded': [1],
        'CompetitionDistance': [500],
        'Promo2': [0]
    })
    
    # predictions = server.predict(sample_data)
    # print(f"Predicted sales: {predictions}")