import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import shap

from models.sequence_models import MultiTaskSequenceModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CybersecuritySequenceDataset(Dataset):
    """PyTorch Dataset that groups events by sequence_id."""
    def __init__(self, df, feature_cols, max_seq_len=50):
        self.max_seq_len = max_seq_len
        self.feature_cols = feature_cols
        
        # We need an encoder for attack types
        self.attack_encoder = LabelEncoder()
        df['attack_encoded'] = self.attack_encoder.fit_transform(df['attack_type'])
        joblib.dump(self.attack_encoder, "saved_models/attack_label_encoder.pkl")
        
        # Group by sequence_id
        logging.info("Grouping data into sequences...")
        grouped = df.groupby('sequence_id')
        
        self.sequences = []
        self.anomaly_labels = []
        self.attack_labels = []
        
        for _, group in grouped:
            seq_features = group[feature_cols].values
            
            # Label for sequence is 1 if any event in it is anomalous
            anomaly_label = group['label'].max()
            
            # Attack label for sequence is the max (which ignores 0 'None' if an attack is present, assuming None is encoded as 0 or we find the non-normal one)
            # Find the first anomalous attack type, else normal
            anomalies = group[group['label'] == 1]
            if len(anomalies) > 0:
                attack_label = anomalies['attack_encoded'].iloc[0]
            else:
                attack_label = group['attack_encoded'].iloc[0]
                
            self.sequences.append(seq_features)
            self.anomaly_labels.append(anomaly_label)
            self.attack_labels.append(attack_label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Pad or truncate
        if len(seq) < self.max_seq_len:
            pad_len = self.max_seq_len - len(seq)
            padding = np.zeros((pad_len, seq.shape[1]))
            seq = np.vstack((padding, seq))
        else:
            seq = seq[-self.max_seq_len:] # take last N events
            
        return {
            'features': torch.tensor(seq, dtype=torch.float32),
            'anomaly_label': torch.tensor(self.anomaly_labels[idx], dtype=torch.float32),
            'attack_label': torch.tensor(self.attack_labels[idx], dtype=torch.long)
        }

class SequenceTrainer:
    def __init__(self, config_path: str = 'config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
    def prepare_data(self):
        logging.info("Loading data for sequence modeling...")
        df = pd.read_csv('data/synthetic_logs_features.csv')
        df.sort_values(by=['entity_id', 'timestamp'], inplace=True)
        
        exclude_cols = ['entity_id', 'timestamp', 'geo_location', 'resource_accessed', 'command_sequence', 
                        'device_fingerprint', 'label', 'attack_type', 'sequence_id', 'source_ip', 
                        'country', 'city', 'entity_type', 'protocol', 'operating_system', 'browser', 
                        'device_type', 'auth_method', 'risk_features', 'attack_encoded']
        
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Simple time split by sequence_id
        split_idx = int(df['sequence_id'].nunique() * 0.8)
        
        train_df = df[df['sequence_id'] <= split_idx].copy()
        test_df = df[df['sequence_id'] > split_idx].copy()
        
        self.train_dataset = CybersecuritySequenceDataset(train_df, self.feature_cols)
        # Re-use the encoder fitted on train
        test_df['attack_encoded'] = self.train_dataset.attack_encoder.transform(test_df['attack_type'])
        
        # Quick manual build of test dataset to avoid refitting encoder
        self.test_dataset = CybersecuritySequenceDataset(test_df, self.feature_cols)
        self.test_dataset.attack_encoder = self.train_dataset.attack_encoder
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        
        self.num_attack_classes = len(self.train_dataset.attack_encoder.classes_)
        logging.info(f"Number of features: {len(self.feature_cols)}")
        logging.info(f"Number of attack classes: {self.num_attack_classes}")
        
    def train_and_evaluate_model(self, model_type: str):
        logging.info(f"\n{'='*50}\nTraining {model_type.upper()} Model\n{'='*50}")
        self.model = MultiTaskSequenceModel(
            input_dim=len(self.feature_cols), 
            hidden_dim=128,
            num_layers=2, 
            num_attack_classes=self.num_attack_classes,
            model_type=model_type
        ).to(self.device)
        
        # Calculate dynamic pos_weight for binary cross entropy
        num_pos = sum(self.train_dataset.anomaly_labels)
        num_neg = len(self.train_dataset) - num_pos
        calculated_weight = max(1.0, float(num_neg) / max(1.0, float(num_pos)))
        pos_weight = torch.tensor([calculated_weight]).to(self.device)
        self.criterion_anomaly = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Calculate multi-class weights
        attack_counts = pd.Series(self.train_dataset.attack_labels).value_counts()
        weights = [1.0] * self.num_attack_classes
        for c in range(self.num_attack_classes):
            if c in attack_counts:
                weights[c] = float(len(self.train_dataset)) / (self.num_attack_classes * attack_counts[c])
        class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        self.criterion_attack = nn.CrossEntropyLoss(weight=class_weights)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.002)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        
        epochs = 15
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                features = batch['features'].to(self.device)
                anomaly_labels = batch['anomaly_label'].unsqueeze(1).to(self.device)
                attack_labels = batch['attack_label'].to(self.device)
                
                self.optimizer.zero_grad()
                anomaly_logits, attack_logits = self.model(features)
                
                loss_anomaly = self.criterion_anomaly(anomaly_logits, anomaly_labels)
                loss_attack = self.criterion_attack(attack_logits, attack_labels)
                
                # Multi-task loss
                loss = loss_anomaly + 0.5 * loss_attack
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            self.scheduler.step()
            if (epoch+1) % 5 == 0:
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(self.train_loader):.4f}")
             
        # Evaluation
        self.model.eval()
        all_anomaly_labels = []
        all_anomaly_preds = []
        all_attack_labels = []
        all_attack_preds = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                features = batch['features'].to(self.device)
                anomaly_labels = batch['anomaly_label']
                attack_labels = batch['attack_label']
                
                anomaly_logits, attack_logits = self.model(features)
                
                anomaly_probs = torch.sigmoid(anomaly_logits).squeeze().cpu().numpy()
                attack_preds = torch.argmax(torch.softmax(attack_logits, dim=1), dim=1).cpu().numpy()
                
                if anomaly_probs.ndim == 0:
                     anomaly_probs = np.expand_dims(anomaly_probs, 0)
                     
                all_anomaly_labels.extend(anomaly_labels.numpy())
                all_anomaly_preds.extend(anomaly_probs)
                all_attack_labels.extend(attack_labels.numpy())
                all_attack_preds.extend(attack_preds)
                
        roc_auc = roc_auc_score(all_anomaly_labels, all_anomaly_preds)
        pr_auc = average_precision_score(all_anomaly_labels, all_anomaly_preds)
        
        logging.info(f"{model_type.upper()} - ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        
        target_names = [str(c) for c in self.test_dataset.attack_encoder.classes_]
        labels = list(range(len(target_names)))
        report = classification_report(all_attack_labels, all_attack_preds, labels=labels, target_names=target_names, zero_division=0, output_dict=True)
        
        # Save model
        torch.save(self.model.state_dict(), f'saved_models/sequence_{model_type}.pt')
        
        return {
            'Model': f"pytorch_{model_type}_sequence",
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1 Score': report['weighted avg']['f1-score'],
            'ROC AUC': roc_auc,
            'PR AUC': pr_auc
        }

    def generate_shap_explanations(self):
        """Uses DeepExplainer to explain the model's predictions on a background set."""
        logging.info("Generating SHAP explanations for explainability...")
        self.model.eval()
        
        # Take a small background sample
        background_data = []
        for i in range(min(100, len(self.train_dataset))):
            background_data.append(self.train_dataset[i]['features'])
        background = torch.stack(background_data).to(self.device)
        
        # Take a small test sample to explain
        test_data = []
        for i in range(min(10, len(self.test_dataset))):
            test_data.append(self.test_dataset[i]['features'])
        test = torch.stack(test_data).to(self.device)
        
        # We need a wrapper to just output the anomaly logits for SHAP
        class AnomalyModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x)[0]
                
        wrapper = AnomalyModelWrapper(self.model)
        
        # DeepExplainer is used for PyTorch models
        try:
            import shap
            explainer = shap.DeepExplainer(wrapper, background)
            shap_values = explainer.shap_values(test)
            logging.info("SHAP explanations generated successfully.")
        except Exception as e:
            logging.warning(f"SHAP generation had an issue (often PyTorch version specifics): {e}")

if __name__ == "__main__":
    trainer = SequenceTrainer()
    trainer.prepare_data()
    
    results = []
    for model_type in ['lstm', 'gru', 'transformer']:
        metrics = trainer.train_and_evaluate_model(model_type)
        results.append(metrics)
        
    results_df = pd.DataFrame(results)
    results_df.to_csv('reports/sequence_evaluation.csv', index=False)
    logging.info("Saved all sequence model evaluation metrics to reports/sequence_evaluation.csv")
    
    # We generate SHAP on the LSTM model as the primary for dashboard representation
    trainer.train_and_evaluate_model('lstm') 
    trainer.generate_shap_explanations()
