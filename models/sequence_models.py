import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiTaskSequenceModel(nn.Module):
    """
    A PyTorch sequence model designed for multi-task learning:
    1. Binary anomaly detection (is the sequence anomalous?)
    2. Multi-class attack type classification (what kind of attack is it?)
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_attack_classes: int, dropout: float = 0.3, model_type: str = 'lstm'):
        super(MultiTaskSequenceModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_type = model_type.lower()
        
        if self.model_type == 'lstm':
            self.seq_layer = nn.LSTM(
                input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, 
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        elif self.model_type == 'gru':
            self.seq_layer = nn.GRU(
                input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, 
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        elif self.model_type == 'transformer':
            self.pos_encoder = PositionalEncoding(input_dim, dropout)
            encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
            self.seq_layer = nn.TransformerEncoder(encoder_layers, num_layers)
            self.fc_transform = nn.Linear(input_dim, hidden_dim) # To match output dims
        else:
            raise ValueError("model_type must be 'lstm', 'gru', or 'transformer'")
        
        # Shared fully connected layer
        self.fc_shared = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        
        # Task 1: Binary Anomaly Detection
        self.fc_anomaly = nn.Linear(hidden_dim // 2, 1)
        
        # Task 2: Attack Classification
        self.fc_attack = nn.Linear(hidden_dim // 2, num_attack_classes)

    def forward(self, x):
        """
        Forward pass.
        x shape: (batch_size, seq_length, input_dim)
        """
        if self.model_type in ['lstm', 'gru']:
            out, _ = self.seq_layer(x)
            last_hidden_state = out[:, -1, :] # Take last time step
        elif self.model_type == 'transformer':
            x = self.pos_encoder(x)
            out = self.seq_layer(x)
            # Use mean pooling for transformer
            pooled_out = out.mean(dim=1) 
            last_hidden_state = F.relu(self.fc_transform(pooled_out))
            
        # Shared representation
        shared_rep = F.relu(self.fc_shared(last_hidden_state))
        shared_rep = self.dropout(shared_rep)
        
        # Logits
        anomaly_logits = self.fc_anomaly(shared_rep)
        attack_logits = self.fc_attack(shared_rep)
        
        return anomaly_logits, attack_logits
