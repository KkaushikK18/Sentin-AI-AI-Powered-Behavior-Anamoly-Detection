import pytest
import pandas as pd
import numpy as np
from data.feature_engineering import FeatureEngineer
from datetime import datetime, timedelta

@pytest.fixture
def sample_data():
    """Provides a small synthetic dataframe for testing."""
    now = datetime.now()
    data = {
        'entity_id': ['E1', 'E1', 'E1', 'E2'],
        'timestamp': [now, now + timedelta(seconds=60), now + timedelta(seconds=4000), now],
        'login_success': [1, 0, 1, 1],
        'session_duration': [100, 0, 200, 300],
        'source_ip': ['1.1.1.1', '1.1.1.1', '2.2.2.2', '3.3.3.3'],
        'geo_location': ['USA, NY', 'USA, NY', 'China, BJ', 'UK, LDN'],
        'country': ['USA', 'USA', 'China', 'UK'],
        'resource_accessed': ['R1', 'R1', 'R2', 'R3'],
        'device_fingerprint': ['Win_Chrome', 'Win_Chrome', 'Mac_Safari', 'Linux_FF'],
        'auth_method': ['Password', 'Password', 'MFA', 'Key'],
        'entity_type': ['User', 'User', 'User', 'Admin'],
        'protocol': ['HTTPS', 'HTTPS', 'SSH', 'SSH'],
        'operating_system': ['Windows', 'Windows', 'macOS', 'Linux'],
        'browser': ['Chrome', 'Chrome', 'Safari', 'Firefox'],
        'device_type': ['Workstation', 'Workstation', 'Workstation', 'Workstation'],
    }
    return pd.DataFrame(data)

def test_extract_temporal_features(sample_data):
    engineer = FeatureEngineer()
    # Mocking init config logic not needed since we pass dataframe directly
    df = engineer.extract_temporal_features(sample_data.copy())
    
    assert 'time_since_last_event' in df.columns
    assert 'failed_login_count_7d' in df.columns
    
    # E1 events: 0, 60s, 4000s
    assert df.loc[0, 'time_since_last_event'] == 0
    assert df.loc[1, 'time_since_last_event'] == 60
    assert df.loc[2, 'time_since_last_event'] == 3940

def test_extract_behavioral_features(sample_data):
    engineer = FeatureEngineer()
    df = engineer.extract_temporal_features(sample_data.copy()) # dependencies
    df = engineer.extract_behavioral_features(df)
    
    assert 'unique_ips_count' in df.columns
    assert 'impossible_travel_indicator' in df.columns
    
    # E1 3rd event has new IP and country, but time is > 3600 (3940s) so indicator is 0
    assert df.loc[2, 'impossible_travel_indicator'] == 0
    
    # E1 has 2 unique IPs by row 2
    assert df.loc[2, 'unique_ips_count'] == 2

def test_pipeline_integration(sample_data):
    engineer = FeatureEngineer()
    df = sample_data.copy()
    
    df = engineer.extract_temporal_features(df)
    df = engineer.extract_behavioral_features(df)
    df = engineer.encode_categorical_features(df)
    df = engineer.normalize_numerical_features(df)
    df = engineer.generate_sequence_ids(df)
    
    assert 'sequence_id' in df.columns
    assert 'entity_type_encoded' in df.columns
    # Check if numerical normalized
    assert np.isclose(df['session_duration'].mean(), 0, atol=1e-5)
