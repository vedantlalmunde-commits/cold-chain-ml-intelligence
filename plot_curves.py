"""
ROC and Precision-Recall Curve Visualization
Plots ROC curve and precision-recall curve for model evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score
)


def plot_roc_and_pr_curves(model, X_test, y_test, model_name="Model"):
    """
    Plot ROC curve and Precision-Recall curve for binary classification.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model with predict_proba method
    X_test : array-like or DataFrame
        Test features
    y_test : array-like or Series
        True test labels
    model_name : str, default="Model"
        Name of the model for plot labels
    """
    
    # Get probability predictions for positive class
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        print("Error: Model does not support probability predictions.")
        print("Using decision function instead...")
        y_pred_proba = model.decision_function(X_test)
    
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot ROC Curve
    axes[0].plot(
        fpr, tpr,
        color='darkorange',
        lw=2.5,
        label=f'{model_name} (AUC = {roc_auc:.3f})'
    )
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    axes[0].set_title('ROC Curve', fontsize=13, fontweight='bold')
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Plot Precision-Recall Curve
    axes[1].plot(
        recall, precision,
        color='green',
        lw=2.5,
        label=f'{model_name} (AP = {avg_precision:.3f})'
    )
    axes[1].axhline(
        y=np.mean(y_test),
        color='red',
        linestyle='--',
        lw=2,
        label=f'Baseline (Prevalence = {np.mean(y_test):.3f})'
    )
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Precision', fontsize=11, fontweight='bold')
    axes[1].set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    axes[1].legend(loc="best", fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Print metrics
    print("\n" + "="*60)
    print("CURVE METRICS")
    print("="*60)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Average Precision Score: {avg_precision:.4f}")
    
    return fig, (fpr, tpr, roc_auc), (precision, recall, avg_precision)


def plot_roc_curve_only(model, X_test, y_test, model_name="Model", save_path=None):
    """
    Plot only ROC curve.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        True test labels
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the figure
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    plt.title('ROC Curve', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    return plt.gcf()


def plot_pr_curve_only(model, X_test, y_test, model_name="Model", save_path=None):
    """
    Plot only Precision-Recall curve.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        True test labels
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the figure
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2.5, label=f'{model_name} (AP = {avg_precision:.3f})')
    plt.axhline(y=np.mean(y_test), color='red', linestyle='--', lw=2, label=f'Baseline (Prevalence = {np.mean(y_test):.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=11, fontweight='bold')
    plt.ylabel('Precision', fontsize=11, fontweight='bold')
    plt.title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    plt.legend(loc="best", fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")
    
    return plt.gcf()


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from train_test_split import split_dataset
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42,
        class_sep=0.8
    )
    
    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Plot curves
    fig, roc_data, pr_data = plot_roc_and_pr_curves(model, X_test, y_test, "Random Forest")
    
    # Save the figure
    plt.savefig('roc_pr_curves.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'roc_pr_curves.png'")
    
    plt.show()
