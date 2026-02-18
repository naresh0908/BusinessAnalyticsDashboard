"""
ML Training Pipeline: Trains models using GridSearchCV for hyperparameter optimization 
and saves them to disk (.pkl files).
Also handles loading them for inference.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix

# Directories
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')
os.makedirs(MODEL_DIR, exist_ok=True)

class MLEngine:
    def __init__(self):
        self.sales_model = None
        self.churn_model = None
        self.encoders = {}
        self.metrics = {}

    def load_data(self):
        """Load processed data."""
        merged_path = os.path.join(DATA_DIR, 'merged_analytics_data.csv')
        if not os.path.exists(merged_path):
            raise FileNotFoundError("Merged data not found. Please run data_pipeline.py first.")
        # Try different encodings if utf-8 fails
        try:
            return pd.read_csv(merged_path, parse_dates=['order_date'])
        except UnicodeDecodeError:
             return pd.read_csv(merged_path, parse_dates=['order_date'], encoding='latin1')

    def train_sales_model(self):
        """Train Optimized Random Forest Regressor for Sales Prediction."""
        print("\n" + "="*50)
        print("Training Optimized Sales Prediction Model...")
        print("="*50)
        
        df = self.load_data()
        
        # Feature Engineering (Robust Encoding)
        # Explicit mapping to match app.py expected feature names
        encoding_map = {
            'category': 'cat',
            'region': 'reg',
            'payment_method': 'pay',
            'customer_segment': 'seg'
        }
        
        encoders = {}
        for col, prefix in encoding_map.items():
            le = LabelEncoder()
            # Ensure all values are strings
            df[col] = df[col].astype(str)
            df[f'{prefix}_encoded'] = le.fit_transform(df[col])
            encoders[prefix] = le
            
        df['order_month_num'] = df['order_date'].dt.month
        df['order_dow'] = df['order_date'].dt.dayofweek
        
        features = ['quantity', 'unit_price', 'discount_percent', 'cat_encoded', 
                    'reg_encoded', 'pay_encoded', 'seg_encoded', 'order_month_num', 'order_dow']
        target = 'total_amount'
        
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameter Tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=1)
        
        print("Starting Grid Search for Sales Model (this may take a minute)...")
        grid_search.fit(X_train, y_train)
        
        model = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")
        
        # Evaluation
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Sales Model - MAE: ₹{mae:.2f}, R2: {r2:.4f}")
        
        # Save artifacts
        joblib.dump(model, os.path.join(MODEL_DIR, 'sales_model.pkl'))
        joblib.dump(encoders, os.path.join(MODEL_DIR, 'sales_encoders.pkl'))
        
        # Save scatter plot data for UI (actual vs predicted sample)
        sample_indices = np.random.choice(len(y_test), min(200, len(y_test)), replace=False)
        self.metrics['sales'] = {
            'mae': mae, 'r2': r2,
            'actual': y_test.iloc[sample_indices].tolist(),
            'predicted': y_pred[sample_indices].tolist()
        }
        joblib.dump(self.metrics['sales'], os.path.join(MODEL_DIR, 'sales_metrics.pkl'))

    def train_churn_model(self):
        """Train Optimized Random Forest Classifier for Churn Prediction."""
        print("\n" + "="*50)
        print("Training Optimized Churn Prediction Model...")
        print("="*50)
        
        df = self.load_data()
        latest = df['order_date'].max()
        
        # Customer-level aggregation
        # Ensure status exists
        if 'status' not in df.columns:
             df['status'] = np.random.choice(['Delivered', 'Cancelled'], size=len(df), p=[0.9, 0.1])

        cust = df.groupby('customer_id').agg(
            recency=('order_date', lambda x: (latest - x.max()).days),
            frequency=('order_id', 'nunique'),
            monetary=('total_amount', 'sum'),
            avg_discount=('discount_percent', 'mean'),
            avg_quantity=('quantity', 'mean'),
            avg_satisfaction=('satisfaction_score', lambda x: x.mean() if 'satisfaction_score' in df.columns else 3),
            cancel_count=('status', lambda x: (x == 'Cancelled').sum())
        ).reset_index()
        
        # Define Churn Target (Recency > 90 days = Churned)
        # Using a fixed threshold is better for business logic than relative
        cust['churned'] = (cust['recency'] > 90).astype(int)
        
        features = ['recency', 'frequency', 'monetary', 'avg_discount', 
                    'avg_quantity', 'avg_satisfaction', 'cancel_count']
        X = cust[features]
        y = cust['churned']
        
        # Handle Imbalance if needed (for now, standard split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Hyperparameter Tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'class_weight': ['balanced', None]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
        
        print("Starting Grid Search for Churn Model...")
        grid_search.fit(X_train, y_train)
        
        model = grid_search.best_estimator_
        print(f"Best Parameters: {grid_search.best_params_}")
        
        # Evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Churn Model - Accuracy: {acc:.4f}")
        
        # Save artifacts
        joblib.dump(model, os.path.join(MODEL_DIR, 'churn_model.pkl'))
        
        self.metrics['churn'] = {
            'accuracy': acc, 'cm': cm, 'report': report,
            'feature_importance': dict(zip(features, model.feature_importances_))
        }
        joblib.dump(self.metrics['churn'], os.path.join(MODEL_DIR, 'churn_metrics.pkl'))

    def run_training_pipeline(self):
        """Run full training pipeline."""
        self.train_sales_model()
        self.train_churn_model()
        print(f"\n✅ All models trained and optimized. Saved to {MODEL_DIR}")

if __name__ == "__main__":
    engine = MLEngine()
    engine.run_training_pipeline()
