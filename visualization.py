import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd

def plot_correlation_heatmap(data):
    """
    Plot a correlation heatmap for the given data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The data to plot the correlation heatmap for
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the heatmap
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw the heatmap
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm',
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    plt.title('Feature Correlation Heatmap', fontsize=16)
    
    return fig

def plot_feature_importance(model, feature_names, model_type='logistic'):
    """
    Plot feature importance for the given model
    
    Parameters:
    -----------
    model : object
        The trained model (LogisticRegression or XGBClassifier)
    feature_names : array-like
        The names of the features
    model_type : str
        The type of model ('logistic' or 'xgboost')
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the feature importance plot
    """
    if model_type == 'logistic':
        # For Logistic Regression, use coefficients as importance
        if hasattr(model, 'coef_'):
            # For multi-class, take the average of absolute coefficients
            if len(model.coef_.shape) > 1 and model.coef_.shape[0] > 1:
                importance = np.mean(np.abs(model.coef_), axis=0)
            else:
                importance = np.abs(model.coef_[0])
        else:
            importance = np.zeros(len(feature_names))
    else:
        # For XGBoost, use feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            importance = np.zeros(len(feature_names))
    
    # Create a dataframe for easier plotting
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot horizontal barplot
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_importance[:15],  # Show only top 15 features
        palette='viridis',
        ax=ax
    )
    
    ax.set_title(f'Top 15 Feature Importance - {model_type.capitalize()}', fontsize=14)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    return fig

def plot_confusion_matrix(cm, classes):
    """
    Plot a confusion matrix
    
    Parameters:
    -----------
    cm : array-like
        The confusion matrix to plot
    classes : array-like
        The class labels
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the confusion matrix plot
    """
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        cm_norm, 
        annot=cm, 
        fmt='d', 
        cmap='Blues',
        square=True,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax
    )
    
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    
    # Rotate x tick labels for better visibility
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_roc_curve(y_true, lr_probs, xgb_probs, classes):
    """
    Plot ROC curves for both models
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    lr_probs : array-like
        Predicted probabilities from Logistic Regression
    xgb_probs : array-like
        Predicted probabilities from XGBoost
    classes : array-like
        Class labels
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the ROC curve plot
    """
    # Convert y_true to one-hot encoding for ROC curve
    y_true_binary = pd.get_dummies(y_true)
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i, cls in enumerate(classes):
        if cls in y_true_binary.columns:
            # Logistic Regression
            fpr_lr, tpr_lr, _ = roc_curve(y_true_binary[cls], lr_probs[:, i])
            roc_auc_lr = auc(fpr_lr, tpr_lr)
            ax.plot(fpr_lr, tpr_lr, label=f'LogReg: {cls} (AUC = {roc_auc_lr:.2f})', linestyle='-')
            
            # XGBoost
            fpr_xgb, tpr_xgb, _ = roc_curve(y_true_binary[cls], xgb_probs[:, i])
            roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
            ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost: {cls} (AUC = {roc_auc_xgb:.2f})', linestyle='--')
    
    # Reference line
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
    
    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves for Each Class', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_prediction_comparison(y_true, lr_preds, xgb_preds):
    """
    Plot a comparison of model predictions vs true values
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    lr_preds : array-like
        Predictions from Logistic Regression
    xgb_preds : array-like
        Predictions from XGBoost
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the prediction comparison plot
    """
    # Create a DataFrame for the comparison
    comparison_df = pd.DataFrame({
        'True': y_true,
        'LogReg': lr_preds,
        'XGBoost': xgb_preds
    })
    
    # Calculate accuracy for each model
    lr_acc = (comparison_df['True'] == comparison_df['LogReg']).mean()
    xgb_acc = (comparison_df['True'] == comparison_df['XGBoost']).mean()
    
    # Calculate where both models are correct, just one is correct, or both are wrong
    both_correct = ((comparison_df['True'] == comparison_df['LogReg']) & 
                    (comparison_df['True'] == comparison_df['XGBoost']))
    just_lr_correct = ((comparison_df['True'] == comparison_df['LogReg']) & 
                       (comparison_df['True'] != comparison_df['XGBoost']))
    just_xgb_correct = ((comparison_df['True'] != comparison_df['LogReg']) & 
                        (comparison_df['True'] == comparison_df['XGBoost']))
    both_wrong = ((comparison_df['True'] != comparison_df['LogReg']) & 
                 (comparison_df['True'] != comparison_df['XGBoost']))
    
    # Calculate percentages
    both_correct_pct = both_correct.mean() * 100
    just_lr_correct_pct = just_lr_correct.mean() * 100
    just_xgb_correct_pct = just_xgb_correct.mean() * 100
    both_wrong_pct = both_wrong.mean() * 100
    
    # Create a figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot a comparison bar chart
    comparison_data = [lr_acc * 100, xgb_acc * 100]
    ax1.bar(['Logistic Regression', 'XGBoost'], comparison_data, color=['cornflowerblue', 'forestgreen'])
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14)
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for i, v in enumerate(comparison_data):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
    
    # Plot a pie chart of prediction overlap
    overlap_data = [both_correct_pct, just_lr_correct_pct, just_xgb_correct_pct, both_wrong_pct]
    labels = ['Both Correct', 'Only LogReg Correct', 'Only XGBoost Correct', 'Both Wrong']
    colors = ['mediumseagreen', 'cornflowerblue', 'forestgreen', 'tomato']
    ax2.pie(overlap_data, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title('Prediction Overlap Analysis', fontsize=14)
    
    plt.tight_layout()
    
    return fig