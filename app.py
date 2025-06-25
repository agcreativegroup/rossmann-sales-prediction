# app.py
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
print("Model exists:", os.path.exists('models/model_25-06-2025-07-27-17-589.pkl'))
print("Scaler exists:", os.path.exists('models/scaler_25-06-2025-11-27-42.pkl'))
MODEL_PATH = os.getenv("MODEL_PATH", "models/model_25-06-2025-07-27-17-589.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler_25-06-2025-12-24-20.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)  
print("Model and scaler loaded successfully")
def preprocess_data(data):
    """
    Preprocess data for prediction
    """
    # Convert date to datetime if it's a string
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['DayOfWeek'] = data['Date'].dt.dayofweek + 1  # 1-7 instead of 0-6
        data['WeekOfYear'] = data['Date'].dt.isocalendar().week
    
    # Create weekend indicator
    data['IsWeekend'] = data.get('DayOfWeek', 1).apply(lambda x: 1 if x >= 6 else 0)
    
    # Handle boolean columns
    bool_columns = ['IsHoliday', 'IsPromo']
    for col in bool_columns:
        if col in data.columns:
            data[col] = data[col].astype(int)
    
    # Fill missing values
    data = data.fillna(0)
    
    return data

def make_prediction(data):
    """
    Make sales prediction
    """
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Preprocess data
        processed_data = preprocess_data(data.copy())
        
        # Select features (adjust based on your model)
        feature_columns = [
            'Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday',
            'SchoolHoliday', 'StoreType', 'Assortment',
            'CompetitionDistance', 'Promo2', 'Year', 'Month', 'Day',
            'WeekOfYear', 'IsWeekend'
        ]
        
        # Keep only available columns
        available_columns = [col for col in feature_columns if col in processed_data.columns]
        X = processed_data[available_columns]
        
        # Scale data
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Make prediction
        predictions = model.predict(X_scaled)
        
        return predictions, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests
    """
    try:
        # Get form data
        store_id = int(request.form.get('store_id', 1))
        date = request.form.get('date')
        is_holiday = int(request.form.get('is_holiday', 0))
        is_weekend = int(request.form.get('is_weekend', 0))
        is_promo = int(request.form.get('is_promo', 0))
        open_store = int(request.form.get('open', 1))
        
        # Create dataframe
        data = pd.DataFrame({
            'Store': [store_id],
            'Date': [date],
            'Open': [open_store],
            'Promo': [is_promo],
            'StateHoliday': [is_holiday],
            'SchoolHoliday': [is_holiday],
            'StoreType': [1],  # Default values
            'Assortment': [1],
            'CompetitionDistance': [1000],
            'Promo2': [0]
        })
        
        # Make prediction
        predictions, error = make_prediction(data)
        
        if error:
            return jsonify({'error': error})
        
        # Estimate customers (simple heuristic)
        estimated_customers = max(int(predictions[0] / 10), 1)
        
        return jsonify({
            'predicted_sales': round(float(predictions[0]), 2),
            'estimated_customers': estimated_customers,
            'store_id': store_id,
            'date': date
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Handle batch prediction from CSV file
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file and file.filename.endswith('.csv'):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read CSV
            data = pd.read_csv(filepath)
            
            # Make predictions
            predictions, error = make_prediction(data)
            
            if error:
                return jsonify({'error': error})
            
            # Add predictions to dataframe
            data['Predicted_Sales'] = predictions
            data['Estimated_Customers'] = (predictions / 10).astype(int)
            
            # Save results
            result_filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            data.to_csv(result_path, index=False)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'message': 'Batch prediction completed',
                'download_url': f'/download/{result_filename}',
                'total_predictions': len(predictions)
            })
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download/<filename>')
def download_file(filename):
    """
    Download prediction results
    """
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/plot')
def plot():
    """
    Generate and return a sample plot
    """
    try:
        # Sample data for demonstration
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        sales = np.random.normal(5000, 1000, 30)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(dates, sales, marker='o', linewidth=2, markersize=4)
        plt.title('Predicted Sales Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sales', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return jsonify({'plot': plot_url})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)