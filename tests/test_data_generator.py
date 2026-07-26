import pytest
import pandas as pd
import os
import yaml
from data.data_generator import BehaviorDataGenerator

def test_data_generator_initialization():
    generator = BehaviorDataGenerator()
    assert len(generator.entities) == 260 # 100 users + 10 service + 150 devices

def test_generate_dataset():
    # Load config and override values for faster test
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    config['data_generation']['num_users'] = 5
    config['data_generation']['num_service_accounts'] = 2
    config['data_generation']['num_devices'] = 5
    config['data_generation']['num_days'] = 2
    config['data_generation']['out_file'] = "data/test_synthetic_logs.csv"
    
    # Temporarily rewrite config for test
    with open('config/test_config.yaml', 'w') as file:
        yaml.dump(config, file)
        
    generator = BehaviorDataGenerator(config_path='config/test_config.yaml')
    df = generator.generate_dataset()
    
    # Assertions
    assert len(df) > 0
    assert 'label' in df.columns
    assert 'attack_type' in df.columns
    
    # Ensure there's a mix of normal and anomalous if events generated enough
    if len(df) > 100:
        assert 1 in df['label'].values
        assert 0 in df['label'].values
        
    # Cleanup
    if os.path.exists('data/test_synthetic_logs.csv'):
        os.remove('data/test_synthetic_logs.csv')
    if os.path.exists('config/test_config.yaml'):
        os.remove('config/test_config.yaml')
