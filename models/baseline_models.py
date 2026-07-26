import pandas as pd
import numpy as np
import logging
import joblib
import yaml
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaselineAnomalyDetector:
    """
    Trains and evaluates unsupervised baseline models (Isolation Forest, One-Class SVM)
    for behavioral anomaly detection.
    """
    def __init__(self, data_path: str = 'data/synthetic_logs_features.csv', config_path: str = 'config/config.yaml'):
        self.data_path = data_path
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.models = {}
        self.feature_cols = []
        # Calculate expected contamination (total anomaly ratio)
        probs = self.config['data_generation']['attack_probabilities']
        self.contamination = sum(probs.values()) 
        # Slightly overestimate contamination for safety
        self.contamination = min(0.5, self.contamination * 1.5) 
        
        Path("saved_models").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)

    def load_and_prepare_data(self):
        """Loads feature data and separates features from labels."""
        logging.info(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # Sort by time/sequence to maintain temporal integrity if doing train/test split
        df.sort_values(by=['entity_id', 'timestamp'], inplace=True)
        
        # Define columns to exclude from training
        exclude_cols = ['entity_id', 'timestamp', 'geo_location', 'resource_accessed', 'command_sequence', 
                        'device_fingerprint', 'label', 'attack_type', 'sequence_id', 'source_ip', 
                        'country', 'city', 'entity_type', 'protocol', 'operating_system', 'browser', 
                        'device_type', 'auth_method', 'risk_features']
        
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Simple time-based train/test split (80/20)
        split_idx = int(len(df) * 0.8)
        self.X_train = df.iloc[:split_idx][self.feature_cols]
        self.y_train = df.iloc[:split_idx]['label']
        self.X_test = df.iloc[split_idx:][self.feature_cols]
        self.y_test = df.iloc[split_idx:]['label']
        
        logging.info(f"Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        
    def train_isolation_forest(self):
        """Trains an Isolation Forest model."""
        logging.info(f"Training Isolation Forest with contamination={self.contamination:.4f}...")
        clf = IsolationForest(
            n_estimators=100, 
            max_samples='auto', 
            contamination=self.contamination, 
            random_state=42, 
            n_jobs=-1
        )
        clf.fit(self.X_train)
        self.models['isolation_forest'] = clf
        
    def train_one_class_svm(self):
        """Trains a One-Class SVM model."""
        # OCSVM can be very slow on large datasets, so we might sample for training
        logging.info("Training One-Class SVM... (using a sample of 10,000 for efficiency)")
        sample_idx = np.random.choice(self.X_train.index, size=min(10000, len(self.X_train)), replace=False)
        X_train_sample = self.X_train.loc[sample_idx]
        
        clf = OneClassSVM(
            kernel='rbf', 
            gamma='scale', 
            nu=self.contamination # nu is an upper bound on fraction of training errors
        )
        clf.fit(X_train_sample)
        self.models['one_class_svm'] = clf

    def evaluate_models(self):
        """Evaluates all trained models on the test set."""
        results = []
        for name, model in self.models.items():
            logging.info(f"Evaluating {name}...")
            
            # Predict returns 1 for inliers, -1 for outliers
            preds = model.predict(self.X_test)
            
            # Convert predictions to 0 (normal), 1 (anomaly) to match our labels
            y_pred = np.where(preds == -1, 1, 0)
            
            # For ROC/PR curves we need scores. In sklearn, decision_function returns > 0 for inliers, < 0 for outliers.
            # We invert it so higher score = more anomalous
            scores = -model.decision_function(self.X_test)
            
            metrics = {
                'Model': name,
                'Precision': precision_score(self.y_test, y_pred, zero_division=0),
                'Recall': recall_score(self.y_test, y_pred, zero_division=0),
                'F1 Score': f1_score(self.y_test, y_pred, zero_division=0),
                'ROC AUC': roc_auc_score(self.y_test, scores),
                'PR AUC': average_precision_score(self.y_test, scores)
            }
            results.append(metrics)
            
            # Log Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            logging.info(f"\nConfusion Matrix for {name}:\n{cm}")
            
        results_df = pd.DataFrame(results)
        results_df.to_csv("reports/baseline_evaluation.csv", index=False)
        logging.info(f"\nEvaluation Results:\n{results_df.to_string()}")
        return results_df

    def save_models(self):
        """Saves trained models to disk."""
        for name, model in self.models.items():
            path = f"saved_models/{name}.pkl"
            joblib.dump(model, path)
            logging.info(f"Saved {name} to {path}")

    def run_pipeline(self):
        """Runs the full training and evaluation pipeline."""
        self.load_and_prepare_data()
        self.train_isolation_forest()
        self.train_one_class_svm()
        self.evaluate_models()
        self.save_models()

if __name__ == "__main__":
    detector = BaselineAnomalyDetector()
    detector.run_pipeline()
