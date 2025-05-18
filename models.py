import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb

def train_logistic_regression(X_train, y_train, X_test, y_test, C=1.0, max_iter=1000, solver='lbfgs'):
    """
    Train a Logistic Regression model for AQI prediction
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        The training features
    y_train : pandas.Series
        The training target
    X_test : pandas.DataFrame
        The testing features
    y_test : pandas.Series
        The testing target
    C : float
        Inverse of regularization strength
    max_iter : int
        Maximum number of iterations
    solver : str
        Algorithm to use for optimization
        
    Returns:
    --------
    model : LogisticRegression
        The trained Logistic Regression model
    results : dict
        Dictionary containing model performance metrics
    """
    # Create and train the model
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        multi_class='multinomial',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    results = evaluate_model(y_test, y_pred, y_prob, model.classes_)
    
    return model, results

def train_xgboost(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, max_depth=6):
    """
    Train an XGBoost model for AQI prediction
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        The training features
    y_train : pandas.Series
        The training target
    X_test : pandas.DataFrame
        The testing features
    y_test : pandas.Series
        The testing target
    n_estimators : int
        Number of boosting rounds
    learning_rate : float
        Boosting learning rate
    max_depth : int
        Maximum tree depth
        
    Returns:
    --------
    model : xgb.XGBClassifier
        The trained XGBoost model
    results : dict
        Dictionary containing model performance metrics
    """
    # XGBoost requires numerical labels, so we need to convert categorical labels
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    import numpy as np
    
    # Create a clean copy of the target variables to ensure they contain valid AQI categories
    valid_categories = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 
                        'Unhealthy', 'Very Unhealthy', 'Hazardous']
    
    # Clean the training data to ensure it only contains valid AQI categories
    y_train_clean = pd.Series([val if val in valid_categories else 'Moderate' for val in y_train])
    
    # Clean the test data similarly
    y_test_clean = pd.Series([val if val in valid_categories else 'Moderate' for val in y_test])
    
    # Create label encoder and transform target variables
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_clean)
    
    # Transform test data
    try:
        y_test_encoded = le.transform(y_test_clean)
    except ValueError as e:
        # In case there are still unseen labels, this is a fallback
        print(f"Warning: {e}. Using default encoding for unseen labels.")
        y_test_encoded = np.zeros(len(y_test_clean), dtype=int)
    
    # Create and train the model
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        objective='multi:softprob',
        random_state=42
    )
    
    # Train on encoded labels
    model.fit(X_train, y_train_encoded)
    
    # Make predictions (these will be encoded)
    y_pred_encoded = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Convert predictions back to original labels for evaluation
    y_pred = le.inverse_transform(y_pred_encoded)
    
    # Store label encoder in the model for future use
    model.label_encoder_ = le
    
    # Calculate metrics
    results = evaluate_model(y_test_clean, y_pred, y_prob, le.classes_)
    
    return model, results

def evaluate_model(y_true, y_pred, y_prob, classes):
    """
    Evaluate a classification model
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    y_prob : array-like
        Predicted probabilities
    classes : array-like
        Class labels
        
    Returns:
    --------
    dict
        Dictionary containing model performance metrics
    """
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate average metrics for multi-class
    avg_precision = precision_score(y_true, y_pred, average='weighted')
    avg_recall = recall_score(y_true, y_pred, average='weighted')
    avg_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Return all metrics
    return {
        'accuracy': acc,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'classes': classes
    }