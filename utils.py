import pandas as pd
import numpy as np
import os

def load_sample_data():
    """
    Load sample AQI data for demonstration purposes
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing sample AQI data
    """
    # Try to load the dataset from the data directory first
    try:
        data_path = 'data/perfect_aqi_dataset.csv'
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
    except Exception as e:
        print(f"Could not load dataset from file: {e}")
        print("Generating synthetic data instead.")
    
    # If file loading fails, generate synthetic data as fallback
    # Create a sample dataset with realistic AQI data features
    np.random.seed(42)
    
    # Number of samples
    n_samples = 1000
    
    # Generate features related to air quality
    pm25 = np.random.gamma(shape=2.0, scale=10.0, size=n_samples)  # PM2.5 levels
    pm10 = pm25 * 1.5 + np.random.normal(0, 5, n_samples)  # PM10 levels
    so2 = np.random.gamma(shape=1.0, scale=5.0, size=n_samples)  # SO2 levels
    no2 = np.random.gamma(shape=1.5, scale=10.0, size=n_samples)  # NO2 levels
    co = np.random.gamma(shape=0.5, scale=0.5, size=n_samples)  # CO levels
    o3 = np.random.gamma(shape=1.0, scale=15.0, size=n_samples)  # O3 levels
    
    # Temperature, humidity and wind features
    temperature = np.random.normal(25, 10, n_samples)  # Temperature in °C
    humidity = np.random.normal(60, 15, n_samples)  # Relative humidity (%)
    wind_speed = np.random.gamma(shape=2.0, scale=2.0, size=n_samples)  # Wind speed in m/s
    
    # Create temporal features
    # Month (1-12)
    month = np.random.randint(1, 13, n_samples)
    # Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)
    season = np.ceil(month / 3) % 4 + 1
    
    # Location type (urban, suburban, rural, industrial)
    location_types = ['Urban', 'Suburban', 'Rural', 'Industrial']
    location_type = np.random.choice(location_types, n_samples)
    
    # Create AQI buckets based on the features
    # Calculate a weighted sum as a proxy for AQI
    aqi_proxy = (pm25 * 3.0 + pm10 * 1.5 + so2 * 2.0 + no2 * 2.0 + 
                co * 10.0 + o3 * 1.0 - wind_speed * 5.0 + 
                np.where(season == 2, 20, 0))  # Summer penalty
    
    # Create AQI buckets
    conditions = [
        (aqi_proxy < 50),
        (aqi_proxy >= 50) & (aqi_proxy < 100),
        (aqi_proxy >= 100) & (aqi_proxy < 150),
        (aqi_proxy >= 150) & (aqi_proxy < 200),
        (aqi_proxy >= 200) & (aqi_proxy < 300),
        (aqi_proxy >= 300)
    ]
    
    aqi_buckets = [
        'Good',
        'Moderate',
        'Unhealthy for Sensitive Groups',
        'Unhealthy',
        'Very Unhealthy',
        'Hazardous'
    ]
    
    aqi_bucket = np.select(conditions, aqi_buckets, default='Unknown')
    
    # Create DataFrame
    data = pd.DataFrame({
        'PM2.5': pm25,
        'PM10': pm10,
        'SO2': so2,
        'NO2': no2,
        'CO': co,
        'O3': o3,
        'Temperature': temperature,
        'Humidity': humidity,
        'Wind_Speed': wind_speed,
        'Month': month,
        'Season': season,
        'Location_Type': location_type,
        'AQI_Bucket': aqi_bucket
    })
    
    # Add some missing values to make it more realistic
    for col in data.columns:
        if col != 'AQI_Bucket':  # Don't add missing values to the target column
            # Add 2% missing values
            mask = np.random.random(n_samples) < 0.02
            data.loc[mask, col] = np.nan
    
    return data

def save_model(model, model_name):
    """
    Save a trained model to disk
    
    Parameters:
    -----------
    model : object
        The trained model to save
    model_name : str
        The name to use for the saved model
    """
    import joblib
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model
    joblib.dump(model, f'models/{model_name}.joblib')
    print(f"Model saved as models/{model_name}.joblib")

def load_model(model_name):
    """
    Load a trained model from disk
    
    Parameters:
    -----------
    model_name : str
        The name of the model to load
        
    Returns:
    --------
    object
        The loaded model
    """
    import joblib
    
    model_path = f'models/{model_name}.joblib'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    # Load the model
    return joblib.load(model_path)