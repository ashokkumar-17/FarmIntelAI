import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Step 1: Data Preparation
def load_data(file_path):
    """
    Load and preprocess market trend data from a CSV file.
    """
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])  # Ensure date column is in datetime format
    data.fillna(method='ffill', inplace=True)  # Forward-fill missing values
    return data

# Step 2: Feature Engineering
def feature_engineering(data):
    """
    Engineer features for model input.
    """
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year

    label_encoder = LabelEncoder()
    data['Crop_Encoded'] = label_encoder.fit_transform(data['Crop'])

    X = data[['Crop_Encoded', 'Month', 'Year', 'Rainfall', 'Temperature', 'Soil_Quality']]
    y = data['Market_Price']

    return X, y, label_encoder

# Step 3: Model Training
def train_model(X, y):
    """
    Train a machine learning model to predict market prices.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"R-squared: {r2:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")

    return model, r2

# Step 4: Prediction
def predict_viability(model, label_encoder, crop, month, year, rainfall, temperature, soil_quality):
    """
    Predict the economic viability of a crop.
    """
    crop_encoded = label_encoder.transform([crop])[0]
    features = np.array([[crop_encoded, month, year, rainfall, temperature, soil_quality]])
    prediction = model.predict(features)
    return prediction[0]

# Load data and train model
data_file = "C:\Users\ashok\Downloads\Agricultural_Market_Trends_India.csv"  
data = load_data(data_file)
X, y, label_encoder = feature_engineering(data)
model, r2 = train_model(X, y)

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    crop = data['crop']
    month = data['month']
    year = data['year']
    rainfall = data['rainfall']
    temperature = data['temperature']
    soil_quality = data['soil_quality']

    try:
        predicted_price = predict_viability(model, label_encoder, crop, month, year, rainfall, temperature, soil_quality)
        return jsonify({"predicted_price": predicted_price})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)