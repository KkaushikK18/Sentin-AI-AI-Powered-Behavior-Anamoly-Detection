import pytest
import torch
from models.sequence_models import MultiTaskSequenceModel

def test_multi_task_sequence_model():
    batch_size = 4
    seq_len = 10
    input_dim = 20
    hidden_dim = 32
    num_layers = 2
    num_classes = 5
    
    model = MultiTaskSequenceModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_attack_classes=num_classes
    )
    
    # Mock input: (batch_size, seq_length, input_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    anomaly_logits, attack_logits = model(x)
    
    # Anomaly output should be (batch_size, 1)
    assert anomaly_logits.shape == (batch_size, 1)
    
    # Attack output should be (batch_size, num_classes)
    assert attack_logits.shape == (batch_size, num_classes)
