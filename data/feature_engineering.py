import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    """
    Pipeline for extracting meaningful ML features from raw cybersecurity behavioral logs.
    Handles temporal, geographical, behavioral, and sequence-based feature extraction.
    """
    def __init__(self, config_path: str = 'config/config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.input_file = self.config['data_generation']['out_file']
        self.output_file = self.input_file.replace('.csv', '_features.csv')
        
        # Label encoders for categorical variables
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        """Loads the raw synthetic logs."""
        logging.info(f"Loading data from {self.input_file}...")
        df = pd.read_csv(self.input_file, parse_dates=['timestamp'])
        df.sort_values(by=['entity_id', 'timestamp'], inplace=True)
        return df

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts time-based features and rolling statistics."""
        logging.info("Extracting temporal features...")
        
        # Time since last event
        df['time_since_last_event'] = df.groupby('entity_id')['timestamp'].diff().dt.total_seconds().fillna(0)
        
        # Rolling windows for past 24 hours (approximated by last N events for simplicity, 
        # or true rolling if indexed by time. Using shift/expanding for grouped data is easier here).
        
        # Failed login count (expanding sum of failures)
        df['is_failure'] = (df['login_success'] == 0).astype(int)
        df['failed_login_count_7d'] = df.groupby('entity_id')['is_failure'].transform(
            lambda x: x.rolling(window=50, min_periods=1).sum() # approximate window
        )
        
        # Login frequency (expanding count)
        df['login_count_overall'] = df.groupby('entity_id').cumcount() + 1
        
        # Session duration moving average
        df['session_duration_ma'] = df.groupby('entity_id')['session_duration'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
        
        return df

    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts features relating to changes in standard behavior."""
        logging.info("Extracting behavioral features...")
        
        # Unique IPs seen so far
        df['unique_ips_count'] = df.groupby('entity_id')['source_ip'].transform(lambda x: (~x.duplicated()).cumsum())
        
        # Unique locations seen so far
        df['unique_locations_count'] = df.groupby('entity_id')['geo_location'].transform(lambda x: (~x.duplicated()).cumsum())
        
        # Unique resources accessed so far
        df['unique_resources_count'] = df.groupby('entity_id')['resource_accessed'].transform(lambda x: (~x.duplicated()).cumsum())
        
        # Device mismatch indicator (did device fingerprint change from the very first one?)
        # We find the most common (mode) device for the user as baseline, or expanding mode
        # For simplicity: check if current device equals the previous device
        df['prev_device'] = df.groupby('entity_id')['device_fingerprint'].shift(1)
        df['device_changed'] = (df['device_fingerprint'] != df['prev_device']) & df['prev_device'].notna()
        df['device_changed'] = df['device_changed'].astype(int)
        
        # Authentication change
        df['prev_auth'] = df.groupby('entity_id')['auth_method'].shift(1)
        df['auth_changed'] = (df['auth_method'] != df['prev_auth']) & df['prev_auth'].notna()
        df['auth_changed'] = df['auth_changed'].astype(int)
        
        # Geo velocity approximation (Did country change in a short time?)
        df['prev_country'] = df.groupby('entity_id')['country'].shift(1)
        df['country_changed'] = (df['country'] != df['prev_country']) & df['prev_country'].notna()
        df['impossible_travel_indicator'] = (df['country_changed'] & (df['time_since_last_event'] < 3600)).astype(int) # Changed country in < 1 hour
        
        df.drop(columns=['is_failure', 'prev_device', 'prev_auth', 'prev_country', 'country_changed'], inplace=True)
        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes string categories to numeric using LabelEncoder."""
        logging.info("Encoding categorical features...")
        categorical_cols = ['entity_type', 'auth_method', 'protocol', 'operating_system', 'browser', 'device_type']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            
        return df

    def normalize_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes numerical features."""
        logging.info("Normalizing numerical features...")
        numeric_cols = ['session_duration', 'time_since_last_event', 'failed_login_count_7d', 
                        'login_count_overall', 'session_duration_ma', 'unique_ips_count', 
                        'unique_locations_count', 'unique_resources_count']
        
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df
        
    def generate_sequence_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assigns a sequence ID for LSTM/Transformer processing."""
        logging.info("Generating sequence IDs...")
        # Group by entity and date to create daily sequences
        df['date'] = df['timestamp'].dt.date
        df['sequence_id'] = df.groupby(['entity_id', 'date']).ngroup()
        df.drop(columns=['date'], inplace=True)
        return df

    def run_pipeline(self):
        """Executes the full feature engineering pipeline."""
        df = self.load_data()
        df = self.extract_temporal_features(df)
        df = self.extract_behavioral_features(df)
        df = self.encode_categorical_features(df)
        df = self.normalize_numerical_features(df)
        df = self.generate_sequence_ids(df)
        
        # Fill NaNs
        df.fillna(0, inplace=True)
        
        # Save processed features
        df.to_csv(self.output_file, index=False)
        logging.info(f"Feature engineering complete. Saved to {self.output_file} with shape {df.shape}.")
        return df

if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.run_pipeline()
