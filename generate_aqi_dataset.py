import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def generate_perfect_aqi_dataset(n_samples=1500, random_seed=42):
    """
    Generate a perfect dataset for AQI prediction with realistic values
    and clear relationships between features and target.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        The generated dataset
    """
    np.random.seed(random_seed)
    
    # Generate features with realistic values and clear patterns for prediction
    
    # Pollution metrics (main determinants of AQI)
    pm25 = np.random.gamma(shape=2.0, scale=15.0, size=n_samples)  # PM2.5 (μg/m³) - Major factor in AQI
    pm10 = pm25 * 1.8 + np.random.normal(0, 5, n_samples)  # PM10 levels (μg/m³)
    so2 = np.random.gamma(shape=1.5, scale=10.0, size=n_samples)  # SO2 levels (ppb)
    no2 = np.random.gamma(shape=2.0, scale=15.0, size=n_samples)  # NO2 levels (ppb)
    co = np.random.gamma(shape=0.8, scale=1.0, size=n_samples)  # CO levels (ppm)
    o3 = np.random.gamma(shape=2.0, scale=20.0, size=n_samples)  # O3 levels (ppb)
    
    # Weather conditions (influence pollutant dispersion)
    temperature = np.random.normal(25, 10, n_samples)  # Temperature in °C
    humidity = np.random.normal(60, 15, n_samples)  # Relative humidity (%)
    wind_speed = np.random.gamma(shape=2.0, scale=2.0, size=n_samples)  # Wind speed in m/s
    precipitation = np.random.exponential(0.5, n_samples)  # Precipitation in mm
    
    # Temporal features
    month = np.random.randint(1, 13, n_samples)  # Month (1-12)
    day_of_week = np.random.randint(1, 8, n_samples)  # Day of week (1-7)
    season = np.ceil(month / 3) % 4 + 1  # Season (1: Spring, 2: Summer, 3: Fall, 4: Winter)
    
    # Geographical and environmental context
    elevation = np.random.normal(300, 150, n_samples)  # Elevation in meters
    location_types = ['Urban', 'Suburban', 'Rural', 'Industrial']
    location_type = np.random.choice(location_types, n_samples, 
                                    p=[0.4, 0.3, 0.2, 0.1])  # Location type
    
    # Traffic and human activity influence
    traffic_density = np.random.gamma(shape=2.0, scale=3.0, size=n_samples)  # Vehicles per hour per lane
    population_density = np.random.gamma(shape=1.5, scale=2000, size=n_samples)  # People per sq km
    
    # Normalize each factor to be within a reasonable range first
    normalized_pm25 = pm25 / 30.0 * 100  # Normalize to 0-100 range approximately
    normalized_pm10 = pm10 / 50.0 * 100
    normalized_so2 = so2 / 15.0 * 100
    normalized_no2 = no2 / 30.0 * 100
    normalized_co = co / 0.8 * 100
    normalized_o3 = o3 / 40.0 * 100
    
    # Calculate a weighted AQI proxy that combines all pollutants in a realistic way
    # Higher values = worse air quality
    # This creates a clear and more balanced relationship between features and target
    aqi_proxy = (
        normalized_pm25 * 0.3 +     # PM2.5 has highest weight in AQI calculation
        normalized_pm10 * 0.2 +     # PM10 is important but less than PM2.5
        normalized_so2 * 0.1 +      # SO2 contribution
        normalized_no2 * 0.1 +      # NO2 contribution
        normalized_co * 0.1 +       # CO contribution
        normalized_o3 * 0.1 -       # O3 contribution
        wind_speed * 5.0 +          # Wind disperses pollutants (negative effect)
        (np.sin((month-1)*np.pi/6) + 1) * 5 +  # Seasonal patterns
        (1 - precipitation/2).clip(0, 1) * 10 +  # Rain cleans the air (negative effect)
        np.where(location_type == 'Industrial', 20, 
                np.where(location_type == 'Urban', 10, 
                        np.where(location_type == 'Suburban', 5, 0))) +  # Location penalty
        traffic_density * 0.5 +     # Traffic increases pollution
        (population_density / 5000).clip(0, 3) * 5  # Population density effect
    )
    
    # Handle negative values that might occur due to wind speed subtraction
    aqi_proxy = np.maximum(aqi_proxy, 0)
    
    # Apply a scaling transformation to get a more balanced distribution
    aqi_proxy = np.sqrt(aqi_proxy) * 10
    
    # Apply quantile-based thresholds to ensure balanced categories
    good_threshold = np.percentile(aqi_proxy, 16.67)  # Bottom 16.67%
    moderate_threshold = np.percentile(aqi_proxy, 33.33)  # Next 16.67%
    usg_threshold = np.percentile(aqi_proxy, 50)  # Next 16.67% 
    unhealthy_threshold = np.percentile(aqi_proxy, 66.67)  # Next 16.67%
    very_unhealthy_threshold = np.percentile(aqi_proxy, 83.33)  # Next 16.67%
    # Top 16.67% will be Hazardous
    
    # Create AQI buckets based on quantile-based thresholds for a balanced dataset
    conditions = [
        (aqi_proxy < good_threshold),  # Good
        (aqi_proxy >= good_threshold) & (aqi_proxy < moderate_threshold),  # Moderate
        (aqi_proxy >= moderate_threshold) & (aqi_proxy < usg_threshold),  # Unhealthy for Sensitive Groups
        (aqi_proxy >= usg_threshold) & (aqi_proxy < unhealthy_threshold),  # Unhealthy
        (aqi_proxy >= unhealthy_threshold) & (aqi_proxy < very_unhealthy_threshold),  # Very Unhealthy
        (aqi_proxy >= very_unhealthy_threshold)  # Hazardous
    ]
    
    aqi_buckets = [
        'Good',
        'Moderate',
        'Unhealthy for Sensitive Groups',
        'Unhealthy',
        'Very Unhealthy',
        'Hazardous'
    ]
    
    # Convert proxy to actual AQI categories
    aqi_bucket = np.select(conditions, aqi_buckets, default='Unknown')
    
    # Calculate numeric AQI value (0-500 scale) for reference
    # Map AQI proxy to standard AQI range, ensuring realistic values for each category
    aqi_value = np.zeros_like(aqi_proxy)
    
    # Good: 0-50
    mask = (aqi_proxy < good_threshold)
    aqi_value[mask] = aqi_proxy[mask] * (50 / good_threshold)
    
    # Moderate: 51-100
    mask = (aqi_proxy >= good_threshold) & (aqi_proxy < moderate_threshold)
    normalized = (aqi_proxy[mask] - good_threshold) / (moderate_threshold - good_threshold)
    aqi_value[mask] = 51 + normalized * 49
    
    # Unhealthy for Sensitive Groups: 101-150
    mask = (aqi_proxy >= moderate_threshold) & (aqi_proxy < usg_threshold)
    normalized = (aqi_proxy[mask] - moderate_threshold) / (usg_threshold - moderate_threshold)
    aqi_value[mask] = 101 + normalized * 49
    
    # Unhealthy: 151-200
    mask = (aqi_proxy >= usg_threshold) & (aqi_proxy < unhealthy_threshold)
    normalized = (aqi_proxy[mask] - usg_threshold) / (unhealthy_threshold - usg_threshold)
    aqi_value[mask] = 151 + normalized * 49
    
    # Very Unhealthy: 201-300
    mask = (aqi_proxy >= unhealthy_threshold) & (aqi_proxy < very_unhealthy_threshold)
    normalized = (aqi_proxy[mask] - unhealthy_threshold) / (very_unhealthy_threshold - unhealthy_threshold)
    aqi_value[mask] = 201 + normalized * 99
    
    # Hazardous: 301-500
    mask = (aqi_proxy >= very_unhealthy_threshold)
    normalized = np.minimum((aqi_proxy[mask] - very_unhealthy_threshold) / (aqi_proxy.max() - very_unhealthy_threshold), 1.0)
    aqi_value[mask] = 301 + normalized * 199
    
    # Clip to standard AQI range and round to integers
    aqi_value = np.round(aqi_value.clip(0, 500))
    
    # Create DataFrame with comprehensive features and clear naming
    data = pd.DataFrame({
        'PM2.5': pm25,                   # Fine particulate matter (μg/m³)
        'PM10': pm10,                    # Coarse particulate matter (μg/m³)
        'SO2': so2,                      # Sulfur dioxide (ppb)
        'NO2': no2,                      # Nitrogen dioxide (ppb)
        'CO': co,                        # Carbon monoxide (ppm)
        'O3': o3,                        # Ozone (ppb)
        'Temperature': temperature,      # Temperature (°C)
        'Humidity': humidity,            # Relative humidity (%)
        'Wind_Speed': wind_speed,        # Wind speed (m/s)
        'Precipitation': precipitation,  # Precipitation (mm)
        'Month': month,                  # Month of year (1-12)
        'Day_Of_Week': day_of_week,      # Day of week (1-7)
        'Season': season,                # Season (1-4)
        'Elevation': elevation,          # Elevation (meters)
        'Location_Type': location_type,  # Type of location
        'Traffic_Density': traffic_density,      # Vehicles per hour per lane
        'Population_Density': population_density, # People per sq km
        'AQI_Value': aqi_value.astype(int),      # Numeric AQI value (0-500)
        'AQI_Bucket': aqi_bucket         # AQI category (target variable)
    })
    
    # Add a small percentage (2%) of missing values to make it more realistic
    # Don't add missing values to the target column
    for col in data.columns:
        if col != 'AQI_Bucket' and col != 'AQI_Value':
            mask = np.random.random(n_samples) < 0.02
            data.loc[mask, col] = np.nan
    
    return data

if __name__ == "__main__":
    # Generate the perfect dataset
    aqi_data = generate_perfect_aqi_dataset(n_samples=1500)
    
    # Create the data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save the dataset to CSV
    aqi_data.to_csv('data/perfect_aqi_dataset.csv', index=False)
    
    # Print dataset statistics
    print(f"Dataset created with {len(aqi_data)} samples")
    print("\nTarget distribution:")
    print(aqi_data['AQI_Bucket'].value_counts())
    
    print("\nFeature summary:")
    print(aqi_data.describe().T[['count', 'mean', 'min', 'max']])
    
    print("\nMissing values summary:")
    missing = aqi_data.isnull().sum()
    print(missing[missing > 0])
    
    print("\nDataset saved to 'data/perfect_aqi_dataset.csv'")