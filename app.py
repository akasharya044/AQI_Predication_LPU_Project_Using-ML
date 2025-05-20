import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing import preprocess_data, handle_missing_values, encode_categorical_features
from models import train_logistic_regression, train_xgboost, evaluate_model
from visualization import (
    plot_correlation_heatmap, 
    plot_feature_importance, 
    plot_confusion_matrix,
    plot_roc_curve,
    plot_prediction_comparison
)
from utils import load_sample_data

# Set page configuration
st.set_page_config(
    page_title="AQI Prediction App",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define app title and description
st.title(" Air Quality Index (AQI) Prediction")
st.markdown("""
 An application that predicts Air Quality Index (AQI) using machine learning models.
 and compare the performance of Logistic Regression and XGBoost models.
""")

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'lr_model' not in st.session_state:
    st.session_state.lr_model = None
if 'xgb_model' not in st.session_state:
    st.session_state.xgb_model = None
if 'lr_results' not in st.session_state:
    st.session_state.lr_results = None
if 'xgb_results' not in st.session_state:
    st.session_state.xgb_results = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'target' not in st.session_state:
    st.session_state.target = None

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a section",
    ["Data Upload or Load Sample Data", "Data Preprocessing", "Model Training & Comparison", "Predictions & Visualization"]
)

# Data Upload & Exploration section
if app_mode == "Data Upload or Load Sample Data":
    st.header("Please Upload or Load Sample Data")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success("Data successfully loaded!")
            except Exception as e:
                st.error(f"Error: {e}")
                
        st.markdown("---")
        st.subheader("Or Use Sample Data")
        if st.button("Load Sample AQI Data"):
            st.session_state.data = load_sample_data()
            st.success("Sample data loaded successfully!")
    
    with col2:
        if st.session_state.data is not None:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.data.head())
            
            st.subheader("Data Information")
            buffer = pd.DataFrame({
                'Column': st.session_state.data.columns,
                'Type': [str(dtype) for dtype in st.session_state.data.dtypes],
                'Non-Null Count': st.session_state.data.count(),
                'Missing Values': st.session_state.data.isnull().sum(),
                'Missing Percentage': round(100 * st.session_state.data.isnull().sum() / len(st.session_state.data), 2)
            })
            st.dataframe(buffer)
            
            st.subheader("Basic Statistics")
            st.dataframe(st.session_state.data.describe().T)
    
    if st.session_state.data is not None:
        st.header("Data Visualization")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Distribution of Numerical Features")
            numeric_cols = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_feature = st.selectbox("Select a numerical feature", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(st.session_state.data[selected_feature], kde=True, ax=ax)
            plt.title(f'Distribution of {selected_feature}')
            plt.xlabel(selected_feature)
            plt.ylabel('Frequency')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Correlation Heatmap")
            if len(numeric_cols) > 1:
                fig = plot_correlation_heatmap(st.session_state.data[numeric_cols])
                st.pyplot(fig)
            else:
                st.info("Need at least 2 numerical columns to create a correlation heatmap.")

# Data Preprocessing section
elif app_mode == "Data Preprocessing":
    st.header("Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("Please upload or load data in the 'Data Upload & Exploration' section first.")
    else:
        st.subheader("Please Select Target Variable")
        target_col = st.selectbox(
            "Please Select the target column for AQI prediction",
            st.session_state.data.columns.tolist(),
            index=st.session_state.data.columns.tolist().index('AQI_Bucket') if 'AQI_Bucket' in st.session_state.data.columns else 0
        )
        
        st.subheader("Feature Selection")
        all_cols = st.session_state.data.columns.tolist()
        default_features = [col for col in all_cols if col != target_col]
        selected_features = st.multiselect(
            "Select the features or parameters to use it for prediction ",
            default_features,
            default=default_features
        )
        
        if len(selected_features) == 0:
            st.warning("Please select at least one feature.")
        else:
            st.session_state.features = selected_features
            st.session_state.target = target_col
            
            st.subheader("Data Cleaning")
            missing_strategy = st.radio(
                "Choose a strategy for handling missing values",
                ["Drop rows with missing values", "Fill missing values with mean/mode"]
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Train-Test Split")
                test_size = st.slider("Test data percentage", 10, 50, 20) / 100
                random_state = st.number_input("Random state", 0, 100, 42)
            
            with col2:
                st.subheader("Feature Scaling")
                do_scaling = st.checkbox("Apply Standard Scaling", value=True)
            
            if st.button("Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    # Extract features and target
                    X = st.session_state.data[selected_features]
                    y = st.session_state.data[target_col]
                    
                    # Handle missing values
                    if missing_strategy == "Drop rows with missing values":
                        X = handle_missing_values(X, strategy='drop')
                        # Make sure y is aligned with X after dropping rows
                        y = y.loc[X.index]
                    else:
                        X = handle_missing_values(X, strategy='fill')
                    
                    # Encode categorical features
                    X = encode_categorical_features(X)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Apply scaling if selected
                    if do_scaling:
                        scaler = StandardScaler()
                        X_train = pd.DataFrame(
                            scaler.fit_transform(X_train),
                            columns=X_train.columns,
                            index=X_train.index
                        )
                        X_test = pd.DataFrame(
                            scaler.transform(X_test),
                            columns=X_test.columns,
                            index=X_test.index
                        )
                    
                    # Save preprocessed data in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.preprocessed_data = {
                        'X': X,
                        'y': y,
                        'scaler': scaler if do_scaling else None
                    }
                    
                    st.success("Data preprocessing completed!")
                    
                    st.subheader("Preprocessed Data Preview")
                    st.write("Training Features")
                    st.dataframe(X_train.head())
                    
                    st.write("Target Distribution")
                    y_counts = y.value_counts().reset_index()
                    y_counts.columns = [target_col, 'Count']
                    
                    fig = px.pie(y_counts, values='Count', names=target_col, title=f'Distribution of {target_col}')
                    st.plotly_chart(fig)

# Model Training & Comparison section
elif app_mode == "Model Training & Comparison":
    st.header("Model Training & Comparison")
    
    if st.session_state.X_train is None or st.session_state.y_train is None:
        st.warning("Please complete the data preprocessing step first.")
    else:
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Logistic Regression Parameters**")
            lr_c = st.number_input("Regularization strength (C)", 0.01, 10.0, 1.0, 0.1)
            lr_max_iter = st.number_input("Maximum iterations", 100, 2000, 1000, 100)
            lr_solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
        
        with col2:
            st.write("**XGBoost Parameters**")
            xgb_n_estimators = st.number_input("Number of estimators", 10, 1000, 100, 10)
            xgb_learning_rate = st.number_input("Learning rate", 0.01, 1.0, 0.1, 0.01)
            xgb_max_depth = st.number_input("Maximum depth", 3, 15, 6, 1)
        
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                # Train Logistic Regression model
                lr_model, lr_results = train_logistic_regression(
                    st.session_state.X_train, 
                    st.session_state.y_train,
                    st.session_state.X_test,
                    st.session_state.y_test,
                    C=lr_c,
                    max_iter=lr_max_iter,
                    solver=lr_solver
                )
                
                # Train XGBoost model
                xgb_model, xgb_results = train_xgboost(
                    st.session_state.X_train, 
                    st.session_state.y_train,
                    st.session_state.X_test,
                    st.session_state.y_test,
                    n_estimators=xgb_n_estimators,
                    learning_rate=xgb_learning_rate,
                    max_depth=xgb_max_depth
                )
                
                # Save models and results in session state
                st.session_state.lr_model = lr_model
                st.session_state.xgb_model = xgb_model
                st.session_state.lr_results = lr_results
                st.session_state.xgb_results = xgb_results
                
                st.success("Models trained successfully!")
        
        if st.session_state.lr_results is not None and st.session_state.xgb_results is not None:
            st.subheader("Model Performance Comparison")
            
            # Create a comparison table
            comparison_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                'Logistic Regression': [
                    st.session_state.lr_results['accuracy'],
                    st.session_state.lr_results['precision'],
                    st.session_state.lr_results['recall'],
                    st.session_state.lr_results['f1']
                ],
                'XGBoost': [
                    st.session_state.xgb_results['accuracy'],
                    st.session_state.xgb_results['precision'],
                    st.session_state.xgb_results['recall'],
                    st.session_state.xgb_results['f1']
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Show the comparison table
            st.dataframe(comparison_df)
            
            # Create a bar chart to visualize the comparison
            fig = px.bar(
                comparison_df, 
                x='Metric', 
                y=['Logistic Regression', 'XGBoost'],
                barmode='group',
                title='Model Performance Comparison'
            )
            st.plotly_chart(fig)
            
            # Show confusion matrices
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Logistic Regression Confusion Matrix")
                lr_cm_fig = plot_confusion_matrix(
                    st.session_state.lr_results['confusion_matrix'],
                    st.session_state.lr_results['classes']
                )
                st.pyplot(lr_cm_fig)
            
            with col2:
                st.subheader("XGBoost Confusion Matrix")
                xgb_cm_fig = plot_confusion_matrix(
                    st.session_state.xgb_results['confusion_matrix'],
                    st.session_state.xgb_results['classes']
                )
                st.pyplot(xgb_cm_fig)
            
            # Show ROC curves
            st.subheader("ROC Curves")
            roc_fig = plot_roc_curve(
                st.session_state.y_test,
                st.session_state.lr_results['y_prob'],
                st.session_state.xgb_results['y_prob'],
                st.session_state.lr_results['classes']
            )
            st.pyplot(roc_fig)
            
            # Feature importance
            st.subheader("Feature Importance")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**Logistic Regression Coefficients**")
                if hasattr(st.session_state.lr_model, 'coef_'):
                    lr_importance = plot_feature_importance(
                        st.session_state.lr_model, 
                        st.session_state.X_train.columns, 
                        model_type='logistic'
                    )
                    st.pyplot(lr_importance)
                else:
                    st.info("Coefficients not available for this model")
            
            with col2:
                st.write("**XGBoost Feature Importance**")
                xgb_importance = plot_feature_importance(
                    st.session_state.xgb_model, 
                    st.session_state.X_train.columns,
                    model_type='xgboost'
                )
                st.pyplot(xgb_importance)

# Predictions & Visualization section
elif app_mode == "Predictions & Visualization":
    st.header("Predictions & Visualization")
    
    if st.session_state.lr_model is None or st.session_state.xgb_model is None:
        st.warning("Please train the models first.")
    else:
        st.subheader("Prediction Comparison")
        
        # Plot prediction comparison
        pred_fig = plot_prediction_comparison(
            st.session_state.y_test,
            st.session_state.lr_results['y_pred'],
            st.session_state.xgb_results['y_pred']
        )
        st.pyplot(pred_fig)
        
        st.subheader("Manual Prediction")
        st.write("Enter values for the features to get a prediction:")
        
        # Create input fields for each feature
        input_data = {}
        
        cols = st.columns(3)
        for i, feature in enumerate(st.session_state.X_train.columns):
            col_idx = i % 3
            with cols[col_idx]:
                min_val = float(st.session_state.X_train[feature].min())
                max_val = float(st.session_state.X_train[feature].max())
                mean_val = float(st.session_state.X_train[feature].mean())
                
                # Check if feature values are mostly integers
                if pd.Series(st.session_state.X_train[feature]).apply(lambda x: x.is_integer()).mean() > 0.9:
                    input_data[feature] = st.number_input(
                        f"{feature}", 
                        float(min_val), 
                        float(max_val), 
                        float(mean_val),
                        step=1.0
                    )
                else:
                    input_data[feature] = st.number_input(
                        f"{feature}", 
                        float(min_val), 
                        float(max_val), 
                        float(mean_val),
                        step=0.1
                    )
        
        if st.button("Predict"):
            # Create a DataFrame with the input values
            input_df = pd.DataFrame([input_data])
            
            # Make predictions with both models
            lr_pred = st.session_state.lr_model.predict(input_df)[0]
            lr_prob = st.session_state.lr_model.predict_proba(input_df)[0]
            
            # For XGBoost, handle the encoded labels
            xgb_pred_encoded = st.session_state.xgb_model.predict(input_df)[0]
            xgb_prob = st.session_state.xgb_model.predict_proba(input_df)[0]
            
            # Use the stored label encoder to convert the prediction back to the original category
            if hasattr(st.session_state.xgb_model, 'label_encoder_'):
                xgb_pred = st.session_state.xgb_model.label_encoder_.inverse_transform([xgb_pred_encoded])[0]
            else:
                # Fallback in case the model doesn't have the label encoder attribute
                xgb_pred = str(xgb_pred_encoded)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Logistic Regression Prediction")
                st.write(f"Predicted AQI Category: **{lr_pred}**")
                
                # Show probabilities
                lr_prob_df = pd.DataFrame({
                    'Category': st.session_state.lr_model.classes_,
                    'Probability': lr_prob
                })
                
                fig = px.bar(
                    lr_prob_df,
                    x='Category',
                    y='Probability',
                    title='Prediction Probabilities'
                )
                st.plotly_chart(fig)
            
            with col2:
                st.subheader("XGBoost Prediction")
                st.write(f"Predicted AQI Category: **{xgb_pred}**")
                
                # Show probabilities
                if hasattr(st.session_state.xgb_model, 'label_encoder_'):
                    # Get the original class labels using the label encoder
                    categories = st.session_state.xgb_model.label_encoder_.classes_
                else:
                    # Fallback in case the model doesn't have the label encoder attribute
                    categories = [str(i) for i in range(len(xgb_prob))]
                
                xgb_prob_df = pd.DataFrame({
                    'Category': categories,
                    'Probability': xgb_prob
                })
                
                fig = px.bar(
                    xgb_prob_df,
                    x='Category',
                    y='Probability',
                    title='Prediction Probabilities'
                )
                st.plotly_chart(fig)
        
        st.subheader("AQI Category Descriptions")
        aqi_categories = {
            "Good": "Air quality is considered satisfactory, and air pollution poses little or no risk.",
            "Moderate": "Air quality is acceptable; however, for some pollutants there may be a moderate health concern for a very small number of people.",
            "Unhealthy for Sensitive Groups": "Members of sensitive groups may experience health effects. The general public is not likely to be affected.",
            "Unhealthy": "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.",
            "Very Unhealthy": "Health warnings of emergency conditions. The entire population is more likely to be affected.",
            "Hazardous": "Health alert: everyone may experience more serious health effects."
        }
        
        for category, description in aqi_categories.items():
            expander = st.expander(category)
            with expander:
                st.write(description)