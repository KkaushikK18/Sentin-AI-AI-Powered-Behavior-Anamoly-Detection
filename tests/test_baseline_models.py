import pytest
import pandas as pd
import numpy as np
import os
from models.baseline_models import BaselineAnomalyDetector

@pytest.fixture
def sample_feature_data():
    """Provides a small synthetic features dataframe for testing."""
    np.random.seed(42)
    # 100 rows, mostly normal (0), a few anomalies (1)
    labels = np.random.choice([0, 1], p=[0.95, 0.05], size=100)
    
    data = {
        'entity_id': ['E' + str(i % 5) for i in range(100)],
        'timestamp': pd.date_range(start='1/1/2026', periods=100, freq='h'),
        'label': labels,
        # Numerical features
        'time_since_last_event': np.random.rand(100),
        'failed_login_count_7d': np.random.rand(100) + labels * 5, # anomalies have higher fails
        'login_count_overall': np.random.rand(100),
        'session_duration_ma': np.random.rand(100),
        # Boolean features
        'device_changed': np.random.choice([0, 1], size=100),
        'auth_changed': np.random.choice([0, 1], size=100),
        'impossible_travel_indicator': labels, # anomalies always trigger this in this test data
        # Excluded cols just to test filtering
        'geo_location': ['Loc' for _ in range(100)],
        'resource_accessed': ['Res' for _ in range(100)]
    }
    return pd.DataFrame(data)

def test_baseline_detector_initialization():
    detector = BaselineAnomalyDetector()
    assert detector.contamination > 0
    assert detector.contamination <= 0.5

def test_baseline_detector_pipeline(sample_feature_data):
    # Temporarily save test data
    test_file = 'data/test_features.csv'
    sample_feature_data.to_csv(test_file, index=False)
    
    detector = BaselineAnomalyDetector(data_path=test_file)
    
    # Test Data Loading
    detector.load_and_prepare_data()
    assert len(detector.feature_cols) == 7 # The 7 valid features we provided
    assert 'label' not in detector.feature_cols
    assert len(detector.X_train) == 80
    assert len(detector.X_test) == 20
    
    # Test Isolation Forest
    detector.train_isolation_forest()
    assert 'isolation_forest' in detector.models
    
    # Test One-Class SVM
    detector.train_one_class_svm()
    assert 'one_class_svm' in detector.models
    
    # Test Evaluation
    results = detector.evaluate_models()
    assert len(results) == 2
    assert 'ROC AUC' in results.columns
    
    # Test Saving
    detector.save_models()
    assert os.path.exists('saved_models/isolation_forest.pkl')
    assert os.path.exists('saved_models/one_class_svm.pkl')
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
    if os.path.exists('saved_models/isolation_forest.pkl'):
        os.remove('saved_models/isolation_forest.pkl')
    if os.path.exists('saved_models/one_class_svm.pkl'):
        os.remove('saved_models/one_class_svm.pkl')
