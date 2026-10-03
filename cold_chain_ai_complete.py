"""
Cold Chain Intelligence AI/ML Module - Enhanced with Evaluation & Visualization
Complete implementation with data generation, model training, evaluation metrics, and diagnostic plots.

Usage:
1. Generate synthetic data: python cold_chain_ai_complete.py --generate-data
2. Train models: python cold_chain_ai_complete.py --train
3. Evaluate models: python cold_chain_ai_complete.py --evaluate
4. Plot diagnostics: python cold_chain_ai_complete.py --plot-diagnostics
5. Run API server: python cold_chain_ai_complete.py --serve
6. Make predictions: python cold_chain_ai_complete.py --predict '{"route":"Coastal",...}'
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from io import StringIO

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, accuracy_score,
    precision_recall_curve, roc_curve, auc, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
import xgboost as xgb

# Plotting Libraries
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns

# FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn


# ===========================
# PART 1: SYNTHETIC DATA GENERATION
# ===========================

class SyntheticDataGenerator:
    """Generate realistic cold chain shipment data."""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed
    
    def generate(self, n_rows=5000):
        """Generate synthetic cold chain shipment data."""
        rows = []
        
        route_options = ["North", "South", "East", "West", "Coastal", "Mountain"]
        product_types = ["Fresh Produce", "Dairy", "Meat", "Seafood", "Pharma", "Bakery"]
        mode_options = ["Road", "Rail", "Air", "Sea"]
        storage_types = ["Refrigerated", "Cold Storage", "Controlled Temp", "Ambient"]
        seasons = ["Spring", "Summer", "Monsoon", "Winter"]
        
        for i in range(n_rows):
            route = np.random.choice(route_options)
            product = np.random.choice(product_types)
            mode = np.random.choice(mode_options)
            storage = np.random.choice(storage_types)
            season = np.random.choice(seasons)
            
            distance_km = np.random.uniform(100, 3500)
            transit_hours = np.random.uniform(6, 240)
            temperature = np.random.uniform(-5, 35)
            humidity = np.random.uniform(20, 95)
            weather_temp = np.random.uniform(-10, 40)
            rainfall_mm = np.random.uniform(0, 120)
            wind_speed = np.random.uniform(5, 70)
            storm_risk = np.random.uniform(0, 1)
            handling_incidents = np.random.poisson(1.2)
            prior_failures = np.random.poisson(0.5)
            historical_delay_rate = np.random.uniform(0.02, 0.4)
            packaging_quality = np.random.uniform(0.4, 1.0)
            inventory_age_days = np.random.uniform(1, 21)
            storage_temp = np.random.uniform(0, 18)
            storage_humidity = np.random.uniform(20, 90)
            
            # Risk factors
            route_risk = {
                "North": 0.2, "South": 0.3, "East": 0.35, 
                "West": 0.25, "Coastal": 0.45, "Mountain": 0.55
            }[route]
            product_risk = {
                "Fresh Produce": 0.55, "Dairy": 0.62, "Meat": 0.7,
                "Seafood": 0.75, "Pharma": 0.4, "Bakery": 0.45
            }[product]
            
            # Calculate component scores
            temp_deviation = abs(temperature - 5)
            humidity_penalty = max(0, humidity - 70) / 30
            weather_penalty = (max(0, weather_temp - 30) / 20) + rainfall_mm / 100 + storm_risk * 0.6
            storage_penalty = abs(storage_temp - 5) / 10 + max(0, storage_humidity - 70) / 30
            handling_penalty = handling_incidents * 0.07
            
            # Spoilage score calculation
            spoilage_score = (
                product_risk * 0.35 +
                temp_deviation / 20 * 0.25 +
                humidity_penalty * 0.12 +
                weather_penalty * 0.15 +
                storage_penalty * 0.18 +
                handling_penalty * 0.10 +
                (1 - packaging_quality) * 0.15
            )
            
            # Delay score calculation
            delay_factor = (historical_delay_rate + route_risk + weather_penalty + 
                          (transit_hours / 250))
            delay_score = (
                delay_factor * 0.5 +
                (distance_km / 3000) * 0.2 +
                (wind_speed / 70) * 0.12 +
                (storm_risk * 0.18)
            )
            
            # Failure score calculation
            failure_score = (
                prior_failures * 0.12 +
                delay_score * 0.4 +
                spoilage_score * 0.3 +
                route_risk * 0.18
            )
            
            # Binary labels
            spoilage_label = 1 if spoilage_score > 0.62 else 0
            delay_label = 1 if delay_score > 0.7 else 0
            failure_label = 1 if failure_score > 0.72 else 0
            
            row = {
                "shipment_id": f"SHIP-{i+1:06d}",
                "route": route,
                "product_type": product,
                "transport_mode": mode,
                "storage_type": storage,
                "season": season,
                "distance_km": distance_km,
                "transit_hours": transit_hours,
                "temperature_c": temperature,
                "humidity_pct": humidity,
                "weather_temp_c": weather_temp,
                "rainfall_mm": rainfall_mm,
                "wind_speed_kmh": wind_speed,
                "storm_risk": storm_risk,
                "handling_incidents": handling_incidents,
                "prior_failures": prior_failures,
                "historical_delay_rate": historical_delay_rate,
                "packaging_quality": packaging_quality,
                "inventory_age_days": inventory_age_days,
                "storage_temp_c": storage_temp,
                "storage_humidity_pct": storage_humidity,
                "spoilage_risk_label": spoilage_label,
                "delay_risk_label": delay_label,
                "failure_risk_label": failure_label,
                "spoilage_score": spoilage_score,
                "delay_score": delay_score,
                "failure_score": failure_score
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df


# ===========================
# PART 2: MODEL TRAINING WITH EVALUATION
# ===========================

class ColdChainModelTrainer:
    """Train ML models for cold chain intelligence with comprehensive evaluation."""
    
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.anomaly_model = None
        self.preprocessor = None
        self.metrics = {}
        self.evaluation_results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_rf = None
        self.y_pred_xgb = None
        self.y_prob_rf = None
        self.y_prob_xgb = None
    
    def train(self, df):
        """Train all models with 80/20 split."""
        print("[INFO] Starting model training with 80/20 train-test split...")
        
        # Prepare features and target
        cat_cols = ["route", "product_type", "transport_mode", "storage_type", "season"]
        target_cols = ["spoilage_risk_label", "delay_risk_label", "failure_risk_label"]
        num_cols = [c for c in df.columns 
                   if c not in cat_cols + target_cols + ["shipment_id"] + 
                      ["spoilage_score", "delay_score", "failure_score"]]
        
        X = df[cat_cols + num_cols]
        y_spoilage = df["spoilage_risk_label"]
        
        # 80/20 split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_spoilage, test_size=0.2, random_state=42, stratify=y_spoilage
        )
        
        print(f"[INFO] Training set size: {len(self.X_train)} samples")
        print(f"[INFO] Testing set size: {len(self.X_test)} samples")
        print(f"[INFO] Class distribution - Train: {np.bincount(self.y_train)}")
        print(f"[INFO] Class distribution - Test: {np.bincount(self.y_test)}")
        
        # Build preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("scaler", StandardScaler())]), num_cols),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), 
                 cat_cols)
            ]
        )
        
        # RandomForest Model
        print("\n[INFO] Training RandomForest model...")
        self.rf_model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
            ))
        ])
        self.rf_model.fit(self.X_train, self.y_train)
        
        self.y_pred_rf = self.rf_model.predict(self.X_test)
        self.y_prob_rf = self.rf_model.predict_proba(self.X_test)[:, 1]
        
        print("[INFO] RandomForest predictions complete")
        
        # XGBoost Model
        print("[INFO] Training XGBoost model...")
        self.xgb_model = Pipeline([
            ("preprocessor", ColumnTransformer(
                transformers=[
                    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                     ("scaler", StandardScaler())]), num_cols),
                    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), 
                     cat_cols)
                ]
            )),
            ("model", xgb.XGBClassifier(
                n_estimators=300, max_depth=7, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
                random_state=42, tree_method='hist'
            ))
        ])
        self.xgb_model.fit(self.X_train, self.y_train)
        
        self.y_pred_xgb = self.xgb_model.predict(self.X_test)
        self.y_prob_xgb = self.xgb_model.predict_proba(self.X_test)[:, 1]
        
        print("[INFO] XGBoost predictions complete")
        
        # Anomaly Detection Model
        print("[INFO] Training Isolation Forest for anomaly detection...")
        X_transformed = self.preprocessor.fit_transform(X)
        self.anomaly_model = IsolationForest(
            n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1
        )
        self.anomaly_model.fit(X_transformed)
        print("[INFO] Anomaly detection model trained")
        
        print("\n[INFO] Model training complete!")
        return self
    
    def evaluate(self):
        """Comprehensive model evaluation."""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        
        # RandomForest Evaluation
        print("\n" + "-"*80)
        print("RANDOMFOREST MODEL EVALUATION")
        print("-"*80)
        
        rf_accuracy = accuracy_score(self.y_test, self.y_pred_rf)
        rf_precision = precision_score(self.y_test, self.y_pred_rf)
        rf_recall = recall_score(self.y_test, self.y_pred_rf)
        rf_f1 = f1_score(self.y_test, self.y_pred_rf)
        rf_auc = roc_auc_score(self.y_test, self.y_prob_rf)
        rf_mcc = matthews_corrcoef(self.y_test, self.y_pred_rf)
        
        print(f"\nAccuracy:  {rf_accuracy:.4f}")
        print(f"Precision: {rf_precision:.4f}")
        print(f"Recall:    {rf_recall:.4f}")
        print(f"F1-Score:  {rf_f1:.4f}")
        print(f"ROC-AUC:   {rf_auc:.4f}")
        print(f"Matthews Correlation Coefficient: {rf_mcc:.4f}")
        
        print("\nConfusion Matrix:")
        rf_cm = confusion_matrix(self.y_test, self.y_pred_rf)
        print(rf_cm)
        print(f"True Negatives:  {rf_cm[0][0]}")
        print(f"False Positives: {rf_cm[0][1]}")
        print(f"False Negatives: {rf_cm[1][0]}")
        print(f"True Positives:  {rf_cm[1][1]}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred_rf, 
                                   target_names=["No Spoilage", "Spoilage Risk"]))
        
        self.evaluation_results['rf'] = {
            'accuracy': rf_accuracy,
            'precision': rf_precision,
            'recall': rf_recall,
            'f1': rf_f1,
            'auc': rf_auc,
            'mcc': rf_mcc,
            'confusion_matrix': rf_cm.tolist()
        }
        
        # XGBoost Evaluation
        print("\n" + "-"*80)
        print("XGBOOST MODEL EVALUATION")
        print("-"*80)
        
        xgb_accuracy = accuracy_score(self.y_test, self.y_pred_xgb)
        xgb_precision = precision_score(self.y_test, self.y_pred_xgb)
        xgb_recall = recall_score(self.y_test, self.y_pred_xgb)
        xgb_f1 = f1_score(self.y_test, self.y_pred_xgb)
        xgb_auc = roc_auc_score(self.y_test, self.y_prob_xgb)
        xgb_mcc = matthews_corrcoef(self.y_test, self.y_pred_xgb)
        
        print(f"\nAccuracy:  {xgb_accuracy:.4f}")
        print(f"Precision: {xgb_precision:.4f}")
        print(f"Recall:    {xgb_recall:.4f}")
        print(f"F1-Score:  {xgb_f1:.4f}")
        print(f"ROC-AUC:   {xgb_auc:.4f}")
        print(f"Matthews Correlation Coefficient: {xgb_mcc:.4f}")
        
        print("\nConfusion Matrix:")
        xgb_cm = confusion_matrix(self.y_test, self.y_pred_xgb)
        print(xgb_cm)
        print(f"True Negatives:  {xgb_cm[0][0]}")
        print(f"False Positives: {xgb_cm[0][1]}")
        print(f"False Negatives: {xgb_cm[1][0]}")
        print(f"True Positives:  {xgb_cm[1][1]}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred_xgb,
                                   target_names=["No Spoilage", "Spoilage Risk"]))
        
        self.evaluation_results['xgb'] = {
            'accuracy': xgb_accuracy,
            'precision': xgb_precision,
            'recall': xgb_recall,
            'f1': xgb_f1,
            'auc': xgb_auc,
            'mcc': xgb_mcc,
            'confusion_matrix': xgb_cm.tolist()
        }
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"RandomForest ROC-AUC: {rf_auc:.4f} | XGBoost ROC-AUC: {xgb_auc:.4f}")
        print(f"RandomForest Accuracy: {rf_accuracy:.4f} | XGBoost Accuracy: {xgb_accuracy:.4f}")
    
    def plot_diagnostics(self, output_dir="plots"):
        """Generate diagnostic plots for model evaluation."""
        if 'rf' not in self.evaluation_results or 'xgb' not in self.evaluation_results:
            self.evaluate()
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n[INFO] Generating diagnostic plots to {output_dir}/")
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. ROC Curves
        print("[INFO] Plotting ROC curves...")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        fpr_rf, tpr_rf, _ = roc_curve(self.y_test, self.y_prob_rf)
        fpr_xgb, tpr_xgb, _ = roc_curve(self.y_test, self.y_prob_xgb)
        
        auc_rf = auc(fpr_rf, tpr_rf)
        auc_xgb = auc(fpr_xgb, tpr_xgb)
        
        ax.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC = {auc_rf:.4f})', linewidth=2.5)
        ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.4f})', linewidth=2.5)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5000)', linewidth=2)
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve - Spoilage Risk Prediction', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved: {output_dir}/roc_curves.png")
        plt.close()
        
        # 2. Precision-Recall Curves
        print("[INFO] Plotting precision-recall curves...")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        precision_rf, recall_rf, _ = precision_recall_curve(self.y_test, self.y_prob_rf)
        precision_xgb, recall_xgb, _ = precision_recall_curve(self.y_test, self.y_prob_xgb)
        
        auc_pr_rf = auc(recall_rf, precision_rf)
        auc_pr_xgb = auc(recall_xgb, precision_xgb)
        
        ax.plot(recall_rf, precision_rf, label=f'RandomForest (AUC = {auc_pr_rf:.4f})', linewidth=2.5)
        ax.plot(recall_xgb, precision_xgb, label=f'XGBoost (AUC = {auc_pr_xgb:.4f})', linewidth=2.5)
        
        baseline = (self.y_test == 1).sum() / len(self.y_test)
        ax.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline Precision = {baseline:.4f}', linewidth=2)
        
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve - Spoilage Risk Prediction', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/precision_recall_curves.png", dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved: {output_dir}/precision_recall_curves.png")
        plt.close()
        
        # 3. Confusion Matrices
        print("[INFO] Plotting confusion matrices...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        rf_cm = confusion_matrix(self.y_test, self.y_pred_rf)
        xgb_cm = confusion_matrix(self.y_test, self.y_pred_xgb)
        
        sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                   xticklabels=['No Spoilage', 'Spoilage'], 
                   yticklabels=['No Spoilage', 'Spoilage'])
        axes[0].set_title('RandomForest Confusion Matrix', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('True Label', fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontweight='bold')
        
        sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                   xticklabels=['No Spoilage', 'Spoilage'],
                   yticklabels=['No Spoilage', 'Spoilage'])
        axes[1].set_title('XGBoost Confusion Matrix', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('True Label', fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved: {output_dir}/confusion_matrices.png")
        plt.close()
        
        # 4. Metrics Comparison
        print("[INFO] Plotting metrics comparison...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        rf_scores = [
            self.evaluation_results['rf']['accuracy'],
            self.evaluation_results['rf']['precision'],
            self.evaluation_results['rf']['recall'],
            self.evaluation_results['rf']['f1'],
            self.evaluation_results['rf']['auc']
        ]
        xgb_scores = [
            self.evaluation_results['xgb']['accuracy'],
            self.evaluation_results['xgb']['precision'],
            self.evaluation_results['xgb']['recall'],
            self.evaluation_results['xgb']['f1'],
            self.evaluation_results['xgb']['auc']
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        ax.bar(x - width/2, rf_scores, width, label='RandomForest', alpha=0.8)
        ax.bar(x + width/2, xgb_scores, width, label='XGBoost', alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (rf, xgb_val) in enumerate(zip(rf_scores, xgb_scores)):
            ax.text(i - width/2, rf + 0.02, f'{rf:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, xgb_val + 0.02, f'{xgb_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved: {output_dir}/metrics_comparison.png")
        plt.close()
        
        # 5. Probability Distribution
        print("[INFO] Plotting prediction probability distributions...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist([self.y_prob_rf[self.y_test == 0], self.y_prob_rf[self.y_test == 1]], 
                    label=['No Spoilage', 'Spoilage'], bins=30, alpha=0.7)
        axes[0].set_xlabel('Prediction Probability', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_title('RandomForest Probability Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].hist([self.y_prob_xgb[self.y_test == 0], self.y_prob_xgb[self.y_test == 1]], 
                    label=['No Spoilage', 'Spoilage'], bins=30, alpha=0.7)
        axes[1].set_xlabel('Prediction Probability', fontweight='bold')
        axes[1].set_ylabel('Frequency', fontweight='bold')
        axes[1].set_title('XGBoost Probability Distribution', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/probability_distributions.png", dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved: {output_dir}/probability_distributions.png")
        plt.close()
        
        print(f"\n[INFO] All diagnostic plots saved to {output_dir}/")
    
    def save_models(self, prefix="models"):
        """Save trained models to files."""
        os.makedirs(prefix, exist_ok=True)
        
        with open(f"{prefix}/rf_model.pkl", "wb") as f:
            pickle.dump(self.rf_model, f)
        
        with open(f"{prefix}/xgb_model.pkl", "wb") as f:
            pickle.dump(self.xgb_model, f)
        
        with open(f"{prefix}/anomaly_model.pkl", "wb") as f:
            pickle.dump(self.anomaly_model, f)
        
        with open(f"{prefix}/preprocessor.pkl", "wb") as f:
            pickle.dump(self.preprocessor, f)
        
        print(f"[INFO] Models saved to {prefix}/ directory")
    
    @staticmethod
    def load_models(prefix="models"):
        """Load trained models from files."""
        with open(f"{prefix}/rf_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        
        with open(f"{prefix}/xgb_model.pkl", "rb") as f:
            xgb_model = pickle.load(f)
        
        with open(f"{prefix}/anomaly_model.pkl", "rb") as f:
            anomaly_model = pickle.load(f)
        
        with open(f"{prefix}/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        
        return rf_model, xgb_model, anomaly_model, preprocessor


# ===========================
# PART 3: PREDICTION ENGINE
# ===========================

class PredictionEngine:
    """Make predictions on shipments."""
    
    def __init__(self, rf_model, xgb_model, anomaly_model, preprocessor):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.anomaly_model = anomaly_model
        self.preprocessor = preprocessor
    
    def score_shipment(self, record: dict):
        """Score a single shipment for risk."""
        try:
            df = pd.DataFrame([record])
            
            # Get probabilities from both models
            rf_prob = self.rf_model.predict_proba(df)[0][1]
            xgb_prob = self.xgb_model.predict_proba(df)[0][1]
            
            # Combined spoilage probability (ensemble average)
            spoilage_probability = float((rf_prob + xgb_prob) / 2.0)
            
            # Anomaly detection
            cat_cols = ["route", "product_type", "transport_mode", "storage_type", "season"]
            num_cols = [c for c in record.keys() if c not in cat_cols and c != "shipment_id"]
            X_prepared = pd.DataFrame([record])[cat_cols + num_cols]
            
            transformed = self.preprocessor.transform(X_prepared)
            anomaly_score = -self.anomaly_model.score_samples(transformed)[0]
            anomaly_threshold = np.quantile(-self.anomaly_model.score_samples(transformed), 0.95)
            anomaly_flag = bool(anomaly_score > anomaly_threshold)
            
            # Delay probability heuristic
            delay_probability = min(1.0, max(0.0, (
                record.get("historical_delay_rate", 0.1) +
                record.get("storm_risk", 0.0) +
                (record.get("distance_km", 0) / 5000) +
                (record.get("rainfall_mm", 0) / 150)
            ) / 1.8))
            
            # Combined risk score
            risk_score = (
                spoilage_probability * 0.55 +
                delay_probability * 0.25 +
                (0.7 if anomaly_flag else 0.0) * 0.20
            )
            
            # Determine risk level
            if risk_score > 0.8:
                risk_level = "Critical"
            elif risk_score > 0.6:
                risk_level = "High"
            elif risk_score > 0.35:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            return {
                "shipment_id": record.get("shipment_id", "UNKNOWN"),
                "spoilage_probability": round(float(spoilage_probability), 4),
                "delay_probability": round(float(delay_probability), 4),
                "anomaly_detected": anomaly_flag,
                "anomaly_score": round(float(anomaly_score), 4),
                "risk_score": round(float(risk_score), 4),
                "risk_level": risk_level,
                "prediction_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def batch_predict(self, records: list):
        """Score multiple shipments."""
        results = []
        for record in records:
            results.append(self.score_shipment(record))
        return results


# ===========================
# PART 4: FASTAPI SERVER
# ===========================

class ShipmentInput(BaseModel):
    shipment_id: str
    route: str
    product_type: str
    transport_mode: str
    storage_type: str
    season: str
    distance_km: float
    transit_hours: float
    temperature_c: float
    humidity_pct: float
    weather_temp_c: float
    rainfall_mm: float
    wind_speed_kmh: float
    storm_risk: float
    handling_incidents: int
    prior_failures: int
    historical_delay_rate: float
    packaging_quality: float
    inventory_age_days: float
    storage_temp_c: float
    storage_humidity_pct: float


def create_api_app(prediction_engine):
    """Create FastAPI application."""
    app = FastAPI(
        title="Cold Chain Intelligence API",
        description="AI/ML API for spoilage risk, anomaly detection, and shipment scoring",
        version="1.0.0"
    )
    
    @app.get("/health")
    def health():
        """Health check endpoint."""
        return {
            "status": "ok",
            "service": "cold-chain-intelligence",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.post("/predict")
    def predict_shipment(payload: ShipmentInput):
        """Predict risk for a single shipment."""
        try:
            result = prediction_engine.score_shipment(payload.dict())
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/batch_predict")
    def batch_predict(payloads: List[ShipmentInput]):
        """Predict risk for multiple shipments."""
        try:
            results = prediction_engine.batch_predict([p.dict() for p in payloads])
            return {"count": len(results), "results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/info")
    def model_info():
        """Get model information."""
        return {
            "models": ["RandomForest", "XGBoost"],
            "anomaly_detection": "IsolationForest",
            "version": "1.0.0",
            "created": datetime.now().isoformat()
        }
    
    return app


# ===========================
# PART 5: MAIN CLI
# ===========================

def main():
    """Main entry point with CLI support."""
    parser = argparse.ArgumentParser(
        description="Cold Chain Intelligence AI/ML Module"
    )
    parser.add_argument("--generate-data", action="store_true", 
                       help="Generate synthetic data")
    parser.add_argument("--train", action="store_true", 
                       help="Train models with 80/20 split")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate trained models on test set")
    parser.add_argument("--plot-diagnostics", action="store_true",
                       help="Generate diagnostic plots (ROC, PR curves, confusion matrix, etc.)")
    parser.add_argument("--serve", action="store_true", 
                       help="Run FastAPI server")
    parser.add_argument("--predict", type=str, 
                       help="Make prediction (JSON string)")
    parser.add_argument("--data-size", type=int, default=5000,
                       help="Number of synthetic records to generate")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port for FastAPI server")
    parser.add_argument("--models-dir", type=str, default="models",
                       help="Directory for model files")
    
    args = parser.parse_args()
    
    # Generate synthetic data
    if args.generate_data:
        print("[INFO] Generating synthetic data...")
        generator = SyntheticDataGenerator()
        df = generator.generate(args.data_size)
        df.to_csv("synthetic_cold_chain_data.csv", index=False)
        print(f"[INFO] Generated {len(df)} records and saved to synthetic_cold_chain_data.csv")
        print(f"[INFO] Data shape: {df.shape}")
        print(f"\n[INFO] Sample data:\n{df.head()}")
    
    # Train models
    elif args.train:
        print("[INFO] Loading data...")
        if not os.path.exists("synthetic_cold_chain_data.csv"):
            print("[ERROR] synthetic_cold_chain_data.csv not found. Run --generate-data first.")
            sys.exit(1)
        
        df = pd.read_csv("synthetic_cold_chain_data.csv")
        
        print("[INFO] Training models with 80/20 split...")
        trainer = ColdChainModelTrainer()
        trainer.train(df)
        trainer.save_models(args.models_dir)
        
        print("\n[INFO] Training Complete!")
    
    # Evaluate models
    elif args.evaluate:
        print("[INFO] Loading data...")
        if not os.path.exists("synthetic_cold_chain_data.csv"):
            print("[ERROR] synthetic_cold_chain_data.csv not found. Run --generate-data first.")
            sys.exit(1)
        
        df = pd.read_csv("synthetic_cold_chain_data.csv")
        
        print("[INFO] Training models...")
        trainer = ColdChainModelTrainer()
        trainer.train(df)
        trainer.save_models(args.models_dir)
        
        print("[INFO] Evaluating models on test set...")
        trainer.evaluate()
    
    # Plot diagnostics
    elif args.plot_diagnostics:
        print("[INFO] Loading data...")
        if not os.path.exists("synthetic_cold_chain_data.csv"):
            print("[ERROR] synthetic_cold_chain_data.csv not found. Run --generate-data first.")
            sys.exit(1)
        
        df = pd.read_csv("synthetic_cold_chain_data.csv")
        
        print("[INFO] Training models...")
        trainer = ColdChainModelTrainer()
        trainer.train(df)
        trainer.save_models(args.models_dir)
        
        print("[INFO] Evaluating models for plotting...")
        trainer.evaluate()
        
        print("[INFO] Generating diagnostic plots...")
        trainer.plot_diagnostics("plots")
    
    # Run API server
    elif args.serve:
        print("[INFO] Loading models...")
        if not os.path.exists(args.models_dir):
            print(f"[ERROR] Models directory '{args.models_dir}' not found. Run --train first.")
            sys.exit(1)
        
        rf_model, xgb_model, anomaly_model, preprocessor = ColdChainModelTrainer.load_models(args.models_dir)
        
        engine = PredictionEngine(rf_model, xgb_model, anomaly_model, preprocessor)
        app = create_api_app(engine)
        
        print(f"[INFO] Starting API server on http://0.0.0.0:{args.port}")
        print("[INFO] Visit http://localhost:8000/docs for interactive API documentation")
        
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    
    # Make prediction
    elif args.predict:
        print("[INFO] Loading models...")
        if not os.path.exists(args.models_dir):
            print(f"[ERROR] Models directory '{args.models_dir}' not found. Run --train first.")
            sys.exit(1)
        
        rf_model, xgb_model, anomaly_model, preprocessor = ColdChainModelTrainer.load_models(args.models_dir)
        engine = PredictionEngine(rf_model, xgb_model, anomaly_model, preprocessor)
        
        try:
            record = json.loads(args.predict)
            result = engine.score_shipment(record)
            print("\n[RESULT]")
            print(json.dumps(result, indent=2))
        except json.JSONDecodeError:
            print("[ERROR] Invalid JSON format")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
