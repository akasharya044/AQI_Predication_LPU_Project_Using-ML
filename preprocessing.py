import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(data, target_col='AQI_Bucket'):
    """
    Preprocess the data for AQI prediction
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The raw data to preprocess
    target_col : str
        The name of the target column
        
    Returns:
    --------
    X : pandas.DataFrame
        The preprocessed features
    y : pandas.Series
        The target variable
    """
    # Create a copy of the data
    df = data.copy()
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle missing values
    X = handle_missing_values(X)
    
    # Encode categorical features
    X = encode_categorical_features(X)
    
    return X, y

def handle_missing_values(df, strategy='fill'):
    """
    Handle missing values in the dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with missing values
    strategy : str
        The strategy to handle missing values ('drop' or 'fill')
        
    Returns:
    --------
    pandas.DataFrame
        The dataframe with missing values handled
    """
    if strategy == 'drop':
        return df.dropna()
    
    # Fill numeric columns with mean
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if not numeric_cols.empty:
        imputer = SimpleImputer(strategy='mean')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def encode_categorical_features(df):
    """
    Encode categorical features in the dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with categorical features
        
    Returns:
    --------
    pandas.DataFrame
        The dataframe with encoded categorical features
    """
    # Get categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if cat_cols.empty:
        return df
    
    df_encoded = df.copy()
    
    # Use one-hot encoding for categorical columns with few values
    for col in cat_cols:
        if df[col].nunique() < 10:  # One-hot encode if number of unique values is small
            one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), one_hot], axis=1)
        else:  # Label encode if number of unique values is large
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
    
    return df_encoded